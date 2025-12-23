# -*- coding: utf-8 -*-
"""
@Description: A股盘中实时筛选 (完整展示版: 严苛+备选全量输出)
@Logic: 
    1. 基础门槛: 动态换手 & 均线多头 & 户均市值>15万
    2. 严苛精选: 基础门槛 + (净利>14% & 估值>1)
    3. 宽松备选: 仅需满足基础门槛
@Output: 按估值比从高到低排序，无折叠全量推送
@RunTime: 建议 11:35 / 14:15
"""
import tushare as ts
import pandas as pd
import datetime
import time
import numpy as np
import talib
import requests
import json
import os
import traceback
from tqdm import tqdm

# ================= 配置区域 =================
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
API_CALL_DELAY = 0.02

# ================= 飞书发送模块 (分段发送) =================
def send_feishu_msg_batch(title, lines_list, batch_size=20):
    """
    分批发送长列表，防止消息过长被拒
    lines_list: 包含所有要发送的文本行
    """
    if not FEISHU_WEBHOOK_URL:
        print(f"【模拟发送】{title}")
        for line in lines_list: print(line)
        return

    beijing_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    current_time = beijing_now.strftime('%m-%d %H:%M')
    
    # 头部信息只发一次
    header = f"【{title}】\n{current_time}\n{'='*20}"
    
    # 如果没有内容
    if not lines_list:
        post_to_feishu(f"{header}\n无符合条件标的")
        return

    # 分块发送
    total = len(lines_list)
    current_chunk = []
    
    # 先发个头
    post_to_feishu(header)
    
    for i, line in enumerate(lines_list):
        current_chunk.append(line)
        # 满一批发送一次
        if len(current_chunk) >= batch_size:
            content = "\n".join(current_chunk)
            post_to_feishu(content)
            current_chunk = []
            time.sleep(0.5) # 防止频率限制
    
    # 发送剩余的
    if current_chunk:
        content = "\n".join(current_chunk)
        post_to_feishu(content)
    
    post_to_feishu(f"(发送完毕，共 {total} 条)")

def post_to_feishu(text):
    headers = {'Content-Type': 'application/json'}
    data = {"msg_type": "text", "content": {"text": text}}
    try:
        requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(data), timeout=15)
    except Exception as e: print(f"飞书发送报错: {e}")

# ================= 基础工具函数 =================
def get_beijing_now():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=8)

def get_last_trade_day_history():
    now = get_beijing_now()
    check_date = now - datetime.timedelta(days=1)
    for _ in range(10):
        date_str = check_date.strftime('%Y%m%d')
        try:
            df = pro.trade_cal(exchange='', start_date=date_str, end_date=date_str)
            if not df.empty and df.iloc[0]['is_open'] == 1: return date_str
        except: pass
        check_date -= datetime.timedelta(days=1)
    return None

def calculate_dynamic_threshold(base_threshold=2.5):
    """动态换手率计算"""
    now = get_beijing_now()
    t_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    t_lunch_start = now.replace(hour=11, minute=30, second=0, microsecond=0)
    t_lunch_end = now.replace(hour=13, minute=0, second=0, microsecond=0)
    t_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    minutes = 0
    if now > t_open:
        if now <= t_lunch_start: minutes = (now - t_open).total_seconds() / 60
        elif now <= t_lunch_end: minutes = 120
        elif now <= t_close: minutes = 120 + (now - t_lunch_end).total_seconds() / 60
        else: minutes = 240
    
    ratio = max(0.05, min(1.0, minutes / 240.0))
    return round(base_threshold * ratio, 2), int(ratio * 100)

# ================= 数据获取函数 =================

def get_realtime_snapshot(stock_basics_df):
    print(">> 获取实时行情...")
    code_map = {code.split('.')[0]: code for code in stock_basics_df['ts_code']}
    code_list = list(code_map.keys())
    realtime_dfs = []
    
    for i in tqdm(range(0, len(code_list), 800), desc="实时下载"):
        try:
            df = ts.get_realtime_quotes(code_list[i : i + 800])
            if df is not None and not df.empty: realtime_dfs.append(df)
            time.sleep(0.3)
        except: continue
            
    if not realtime_dfs: return pd.DataFrame()
    full = pd.concat(realtime_dfs, ignore_index=True)
    
    full['price'] = pd.to_numeric(full['price'], errors='coerce')
    full['pre_close'] = pd.to_numeric(full['pre_close'], errors='coerce')
    full['volume'] = pd.to_numeric(full['volume'], errors='coerce')
    full = full[full['price'] > 0].copy()
    full['ts_code'] = full['code'].map(code_map)
    
    if 'name' in full.columns: full = full.drop(columns=['name'])
    
    merged = pd.merge(full, stock_basics_df[['ts_code', 'name', 'float_share', 'total_share', 'industry']], on='ts_code', how='inner')
    merged['turnover_rate_now'] = (merged['volume'] / (merged['float_share'] * 10000)) * 100
    merged['total_mv_now'] = merged['total_share'] * merged['price']
    
    return merged

def get_financial_data(ts_code, trade_date):
    try:
        df_ind = pro.fina_indicator(ts_code=ts_code, limit=2, fields='end_date,profit_dedt')
        df_income = pro.income(ts_code=ts_code, limit=2, fields='end_date,revenue')
        if df_income.empty or df_ind.empty: return None
        
        rev = df_income.iloc[0]['revenue']
        prof_dedt = df_ind.iloc[0]['profit_dedt']
        if pd.isna(rev) or rev == 0: return None
        
        margin = prof_dedt / rev
        return {'revenue': rev, 'profit_dedt': prof_dedt, 'deducted_net_profit_margin': margin}
    except: return None

def get_holders_data(ts_code, trade_date):
    try:
        start_dt = (pd.to_datetime(trade_date) - datetime.timedelta(days=365)).strftime('%Y%m%d')
        df_h = pro.stk_holdernumber(ts_code=ts_code, start_date=start_dt)
        if df_h.empty: return None
        holder_num = df_h.sort_values('end_date', ascending=False).iloc[0]['holder_num']
        
        df_top10 = pro.top10_floatholders(ts_code=ts_code, start_date=start_dt)
        top10_sum = 0
        if not df_top10.empty:
            latest_date = df_top10['end_date'].max()
            top10_sum = df_top10[df_top10['end_date'] == latest_date]['hold_amount'].sum()
            
        return {'holder_num': holder_num, 'top10_shares': top10_sum}
    except: return None

def classify_growth(ts_code):
    try:
        df = pro.income(ts_code=ts_code, limit=5, fields='end_date,revenue')
        df_prof = pro.fina_indicator(ts_code=ts_code, limit=5, fields='end_date,profit_dedt')
        if len(df) < 2 or len(df_prof) < 2: return "数据不足"
        rev_grow = df.iloc[0]['revenue'] > df.iloc[1]['revenue']
        prof_grow = df_prof.iloc[0]['profit_dedt'] > df_prof.iloc[1]['profit_dedt']
        if rev_grow and prof_grow: return "双增长"
        if rev_grow: return "营收增"
        if prof_grow: return "净利增"
        return "双降"
    except: return "未知"

def get_concept(ts_code):
    try:
        df = pro.concept_detail(ts_code=ts_code)
        if not df.empty:
            concepts = df['concept_name'].unique()[:3]
            return ",".join(concepts)
    except: pass
    return "-"

def format_stock_card(row, prefix=""):
    """生成详细卡片"""
    code = row['ts_code'].split('.')[0]
    name = row['name']
    ind = row.get('industry', '-')
    concept = row.get('concept', '-')
    price = row['price']
    
    pre = float(row['pre_close'])
    pct_now = (price - pre) / pre * 100 if pre > 0 else 0
    
    day7 = row.get('day_7_chg', 0)
    turn = row.get('turnover_rate_now', 0)
    val = row.get('valuation_ratio', 0)
    mv_per = row.get('per_capita_mv_wan', 0)
    growth = row.get('growth', '-')
    
    pct_icon = "🔺" if pct_now > 0 else "💚"
    
    card = (
        f"\n{prefix} **{name}** ({code}) | {ind} | {growth}\n"
        f"   {pct_icon} {price:.2f} ({pct_now:+.1f}%) | 换手: {turn:.1f}% | 7日: {day7:+.1f}%\n"
        f"   估值: {val:.2f} | 户均: {mv_per:.1f}万 | 概念: {concept}"
    )
    return card

# ================= 核心逻辑 =================

def run_intraday_screener():
    print(">>> 启动盘中全量筛选...")
    
    last_trade_day = get_last_trade_day_history()
    if not last_trade_day: return
    
    BASE_THRESHOLD = 2.5
    dynamic_threshold, progress = calculate_dynamic_threshold(BASE_THRESHOLD)
    print(f"进度: {progress}%, 阈值: >{dynamic_threshold}%")

    print("获取基础数据...")
    df_basic = pro.daily_basic(trade_date=last_trade_day, fields='ts_code,float_share,total_share')
    df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    df_base = pd.merge(df_basic, df_names, on='ts_code')
    
    df_real = get_realtime_snapshot(df_base)
    if df_real.empty: 
        post_to_feishu("错误: 无法获取实时行情")
        return
        
    df_pass1 = df_real[df_real['turnover_rate_now'] > dynamic_threshold].copy()
    print(f"Step 1 通过: {len(df_pass1)} 只")
    
    if df_pass1.empty:
        post_to_feishu(f"【盘中筛选】\n无股票满足换手率 > {dynamic_threshold}%")
        return

    final_candidates = []
    # 扩大处理范围到 500 只，尽可能不漏
    process_list = df_pass1.sort_values('turnover_rate_now', ascending=False).head(500).to_dict('records')
    
    print("技术面筛选 (Step 2/5/6)...")
    start_date_hist = (pd.to_datetime(last_trade_day) - datetime.timedelta(days=300)).strftime('%Y%m%d')
    
    for row in tqdm(process_list):
        ts_code = row['ts_code']
        curr_p = float(row['price'])
        
        try:
            df_hist = pro.daily(ts_code=ts_code, start_date=start_date_hist, end_date=last_trade_day)
        except: continue
        
        if df_hist is None or len(df_hist) < 120: continue
        df_hist = df_hist.sort_values('trade_date', ascending=True)
        
        closes = df_hist['close'].values.tolist()
        closes.append(curr_p)
        arr = np.array(closes)
        
        try:
            ma20 = talib.SMA(arr, 20)[-1]
            ma60 = talib.SMA(arr, 60)[-1]
            ma120 = talib.SMA(arr, 120)[-1]
        except: continue
        
        # Step 2
        if not (curr_p > ma20 and curr_p > ma60 and curr_p > ma120): continue
            
        # Step 6
        try:
            prev_close = closes[-2]
            arr_prev = np.array(closes[:-1])
            ma20_prev = talib.SMA(arr_prev, 20)[-1]
            ma60_prev = talib.SMA(arr_prev, 60)[-1]
            ma120_prev = talib.SMA(arr_prev, 120)[-1]
            
            if not ((prev_close <= ma20_prev) or (prev_close <= ma60_prev) or (prev_close <= ma120_prev)):
                continue
        except: continue
        
        # Step 5 (Loose)
        try:
            if len(closes) > 20: 
                d20 = closes[-21]
                if d20 >= 1.3 * curr_p: continue
        except: pass
        
        day7_chg = 0.0
        if len(closes) >= 9:
            c7 = closes[-9]
            day7_chg = (curr_p - c7) / c7 * 100
        
        row['day_7_chg'] = day7_chg
        final_candidates.append(row)

    print(f"技术面通过: {len(final_candidates)} 只")
    
    if not final_candidates:
        post_to_feishu("【盘中筛选】\n技术面筛选后无结果")
        return

    print("财务与户均筛选...")
    strict_list = [] 
    loose_list = []  
    
    for row in tqdm(final_candidates):
        ts_code = row['ts_code']
        curr_p = row['price']
        
        holders = get_holders_data(ts_code, last_trade_day)
        if not holders: continue
        
        float_shares_real = row['float_share'] * 10000 
        retail_shares = float_shares_real - holders['top10_shares']
        retail_holders = holders['holder_num'] - 10
        if retail_holders <= 0: continue
        
        per_capita_mv = (retail_shares * curr_p) / retail_holders
        # 核心备选条件：户均 > 15万
        if per_capita_mv <= 150000: continue
        
        row['per_capita_mv_wan'] = per_capita_mv / 10000.0
        row['growth'] = classify_growth(ts_code)
        row['concept'] = get_concept(ts_code)
        
        fin = get_financial_data(ts_code, last_trade_day)
        is_strict = False
        val_ratio = 0
        
        if fin:
            revenue_wan = fin['revenue'] / 10000.0
            if pd.notna(row['total_mv_now']) and row['total_mv_now'] > 0:
                val_ratio = (revenue_wan * (fin['deducted_net_profit_margin'] / 0.14) * 10) / row['total_mv_now']
            
            row['valuation_ratio'] = val_ratio
            if fin['deducted_net_profit_margin'] > 0.14 and val_ratio > 1:
                is_strict = True
        else:
            row['valuation_ratio'] = 0 
        
        if is_strict:
            strict_list.append(row)
        else:
            loose_list.append(row)

    # === 生成最终消息 ===
    # 按照用户要求：全部按照“估值比”从高到低排序
    all_msg_lines = []
    
    all_msg_lines.append(f"进度:{progress}% | 阈值:>{dynamic_threshold}%")
    
    # 1. 严苛精选 (Sort by Valuation Desc)
    if strict_list:
        df_strict = pd.DataFrame(strict_list).sort_values('valuation_ratio', ascending=False)
        all_msg_lines.append(f"\n🏆 **严苛精选 (共{len(df_strict)}只)**")
        for _, row in df_strict.iterrows():
            all_msg_lines.append(format_stock_card(row, "🌟"))
    else:
        all_msg_lines.append("\n⚠️ 严苛组无结果")

    # 2. 宽松备选 (Sort by Valuation Desc)
    if loose_list:
        df_loose = pd.DataFrame(loose_list).sort_values('valuation_ratio', ascending=False)
        all_msg_lines.append(f"\n💡 **宽松备选 (共{len(df_loose)}只)**")
        all_msg_lines.append("   (户均达标 + 技术达标)")
        for _, row in df_loose.iterrows():
            all_msg_lines.append(format_stock_card(row, "🔸"))
    else:
        all_msg_lines.append("\n❌ 备选组无结果")

    # 保存CSV备查
    all_res = strict_list + loose_list
    if all_res:
        df_all = pd.DataFrame(all_res)
        fname = f"Intraday_Full_{datetime.datetime.now().strftime('%H%M')}.csv"
        df_all.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f"CSV生成: {fname}")

    # 发送 (分批)
    send_feishu_msg_batch("A股盘中突围 (Script 1)", all_msg_lines)
    print("执行完毕。")

if __name__ == "__main__":
    try:
        run_intraday_screener()
    except Exception as e:
        traceback.print_exc()
        # 报错也用 post_to_feishu 简单发送
        header = {'Content-Type': 'application/json'}
        d = {"msg_type": "text", "content": {"text": f"盘中脚本报错: {str(e)}"}}
        requests.post(FEISHU_WEBHOOK_URL, headers=header, data=json.dumps(d))
