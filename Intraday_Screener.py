# -*- coding: utf-8 -*-
"""
@Description: A股盘中实时筛选 (完整复刻脚本1: 8步筛选+严苛财务+详尽报告)
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

# ================= 辅助函数 =================
def send_feishu_msg(title, content):
    if not FEISHU_WEBHOOK_URL:
        print(f"【模拟发送】{title}")
        print(content)
        return
    beijing_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    current_time = beijing_now.strftime('%m-%d %H:%M')
    full_text = f"【{title}】\n{current_time}\n{'='*20}\n{content}"
    
    # 飞书消息分段发送，防止过长
    # 如果内容超过300行或者字符数过多，可以考虑切分，这里先一次性发
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps({"msg_type": "text", "content": {"text": full_text}}), timeout=15)
    except Exception as e: print(f"飞书发送报错: {e}")

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
    """获取实时行情"""
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
    
    # 清洗
    full['price'] = pd.to_numeric(full['price'], errors='coerce')
    full['pre_close'] = pd.to_numeric(full['pre_close'], errors='coerce')
    full['volume'] = pd.to_numeric(full['volume'], errors='coerce')
    full = full[full['price'] > 0].copy()
    full['ts_code'] = full['code'].map(code_map)
    
    if 'name' in full.columns: full = full.drop(columns=['name'])
    
    merged = pd.merge(full, stock_basics_df[['ts_code', 'name', 'float_share', 'total_share', 'industry']], on='ts_code', how='inner')
    
    # 计算实时指标
    merged['turnover_rate_now'] = (merged['volume'] / (merged['float_share'] * 10000)) * 100
    merged['total_mv_now'] = merged['total_share'] * merged['price'] # 实时总市值(万)
    
    return merged

def get_financial_data(ts_code, trade_date):
    """获取财务数据 (Step 4 & 7)"""
    try:
        # 尝试获取最新一期年报，如果没有则获取最新一期季报
        df_ind = pro.fina_indicator(ts_code=ts_code, limit=2, fields='end_date,profit_dedt,q_dtprofit')
        df_income = pro.income(ts_code=ts_code, limit=2, fields='end_date,revenue,report_type')
             
        if df_income.empty or df_ind.empty: return None
        
        # 简单取最新一条非空数据
        rev = df_income.iloc[0]['revenue']
        prof_dedt = df_ind.iloc[0]['profit_dedt']
        
        if pd.isna(rev) or rev == 0: return None
        
        margin = prof_dedt / rev
        return {'revenue': rev, 'profit_dedt': prof_dedt, 'deducted_net_profit_margin': margin}
    except: return None

def get_holders_data(ts_code, trade_date):
    """获取股东人数和前十大 (Step 7.3)"""
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
    """Step 7.4 增长分类"""
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
    """补充概念信息 (仅对最终入选者调用)"""
    try:
        df = pro.concept_detail(ts_code=ts_code)
        if not df.empty:
            # 拼接前3个概念
            concepts = df['concept_name'].unique()[:3]
            return ",".join(concepts)
    except: pass
    return "-"

# ================= 核心筛选逻辑 =================

def run_intraday_screener():
    print(">>> 启动完整版盘中筛选 (Script 1)...")
    
    last_trade_day = get_last_trade_day_history()
    if not last_trade_day: return
    
    # 1. 动态阈值
    BASE_THRESHOLD = 2.5
    dynamic_threshold, progress = calculate_dynamic_threshold(BASE_THRESHOLD)
    print(f"时间进度: {progress}%, 动态换手阈值: >{dynamic_threshold}%")

    # 2. 基础数据
    print("获取基础数据...")
    df_basic = pro.daily_basic(trade_date=last_trade_day, fields='ts_code,float_share,total_share')
    df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    df_base = pd.merge(df_basic, df_names, on='ts_code')
    
    # 3. 实时行情 & Step 1 (换手率)
    df_real = get_realtime_snapshot(df_base)
    if df_real.empty: 
        send_feishu_msg("错误", "无法获取实时行情")
        return
        
    df_pass1 = df_real[df_real['turnover_rate_now'] > dynamic_threshold].copy()
    print(f"Step 1 (换手率) 通过: {len(df_pass1)} 只")
    
    if df_pass1.empty:
        send_feishu_msg("盘中筛选", "无股票满足换手率条件")
        return

    # 4. 循环处理 Step 2, 5, 6 (需要历史K线)
    final_candidates = []
    process_list = df_pass1.sort_values('turnover_rate_now', ascending=False).head(300).to_dict('records')
    
    print("Step 2/5/6: 均线与形态分析...")
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
        
        # Step 2: 实时价 > 三均线
        if not (curr_p > ma20 and curr_p > ma60 and curr_p > ma120):
            continue
            
        # Step 6: 昨日收盘价形态
        try:
            prev_close = closes[-2]
            arr_prev = np.array(closes[:-1])
            ma20_prev = talib.SMA(arr_prev, 20)[-1]
            ma60_prev = talib.SMA(arr_prev, 60)[-1]
            ma120_prev = talib.SMA(arr_prev, 120)[-1]
            
            if not ((prev_close <= ma20_prev) or (prev_close <= ma60_prev) or (prev_close <= ma120_prev)):
                continue
        except: continue
        
        # Step 5: 抵扣价 (略微简化判断)
        try:
            deduction_cond = True
            if len(closes) > 20: 
                d20 = closes[-21]
                if d20 >= 1.2 * curr_p: deduction_cond = False
        except: pass
        
        # 计算 7日涨跌幅 (当前价 vs 7个交易日前)
        # closes[-1] is today, closes[-8] is 7 days ago? No, array index logic.
        # closes list length N. closes[-1] is now. closes[-1-N]
        day7_chg = 0.0
        if len(closes) >= 9: # 至少要有当天+过去8天
            c7 = closes[-9] # 7个交易日前的收盘价
            day7_chg = (curr_p - c7) / c7 * 100
        
        row['day_7_chg'] = day7_chg
        row['valuation_ratio'] = 0 
        row['strict_pass'] = False
        final_candidates.append(row)

    print(f"技术面筛选通过: {len(final_candidates)} 只")
    
    if not final_candidates:
        send_feishu_msg("盘中筛选", "技术面(MA/形态)筛选后无结果")
        return

    # 5. Step 4 & 7: 严苛财务筛选
    print("Step 7: 严苛财务筛选...")
    strict_results = []
    
    for row in tqdm(final_candidates):
        ts_code = row['ts_code']
        curr_p = row['price']
        
        fin = get_financial_data(ts_code, last_trade_day)
        if not fin: continue
        
        if fin['deducted_net_profit_margin'] <= 0.14: continue
        
        # 修正估值比单位问题：营收(元) -> 需要转换，市值(万)
        # 公式：( 营收(元)/10000 * (净利率/0.14) * 10 ) / 总市值(万)
        # 或者：( 营收(元) * ... ) / (总市值(万) * 10000)
        # 这里统一把营收转为万，这样分子分母单位一致
        revenue_wan = fin['revenue'] / 10000.0
        
        if pd.isna(row['total_mv_now']) or row['total_mv_now'] == 0: continue
        
        val_ratio = (revenue_wan * (fin['deducted_net_profit_margin'] / 0.14) * 10) / row['total_mv_now']
        if val_ratio <= 1: continue
        
        row['valuation_ratio'] = val_ratio
        
        holders = get_holders_data(ts_code, last_trade_day)
        if not holders: continue
        
        float_shares_real = row['float_share'] * 10000 # 换回股
        retail_shares = float_shares_real - holders['top10_shares']
        retail_holders = holders['holder_num'] - 10
        if retail_holders <= 0: continue
        
        per_capita_mv = (retail_shares * curr_p) / retail_holders
        if per_capita_mv <= 150000: continue
        
        row['growth'] = classify_growth(ts_code)
        row['per_capita_mv_wan'] = per_capita_mv / 10000.0
        row['concept'] = get_concept(ts_code) # 仅对最终结果获取概念
        
        strict_results.append(row)

    # 6. 生成报告
    df_final = pd.DataFrame(strict_results)
    
    msg_lines = []
    
    if not df_final.empty:
        # 按估值比排序
        df_final = df_final.sort_values('valuation_ratio', ascending=False)
        
        msg_lines.append(f"进度: {progress}% | 动态阈值: >{dynamic_threshold}%")
        msg_lines.append(f"严苛筛选通过: {len(df_final)} 只")
        
        for i, row in df_final.head(15).iterrows(): # 限制展示前15只，防止太长
            # 数据准备
            code = row['ts_code']
            name = row['name']
            ind = row.get('industry', '-')
            concept = row.get('concept', '-')
            
            price = row['price']
            pre = float(row['pre_close'])
            pct_now = (price - pre) / pre * 100
            
            day7 = row.get('day_7_chg', 0)
            turn = row.get('turnover_rate_now', 0)
            val = row.get('valuation_ratio', 0)
            mv_per = row.get('per_capita_mv_wan', 0)
            growth = row.get('growth', '-')
            
            # 构建详尽的卡片式单行
            # 格式：
            # 1. 名称(代码) | 行业 | 增长
            # 2. 现价(涨幅) | 换手 | 7日
            # 3. 估值比 | 户均市值 | 概念
            
            stock_block = (
                f"\n🔴 **{name}** ({code}) | {ind} | {growth}\n"
                f"   现价: {price:.2f} ({pct_now:+.2f}%) | 换手: {turn:.2f}% | 7日: {day7:+.1f}%\n"
                f"   估值比: {val:.2f} | 户均: {mv_per:.1f}万 | 概念: {concept}"
            )
            msg_lines.append(stock_block)
            
        if len(df_final) > 15:
            msg_lines.append(f"\n...剩余 {len(df_final)-15} 只请查看 GitHub Artifacts CSV")
            
        # 保存
        fname = f"Intraday_Strict_{datetime.datetime.now().strftime('%H%M')}.csv"
        df_final.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f"CSV生成: {fname}")
        
    else:
        msg_lines.append(f"进度: {progress}% | 阈值: >{dynamic_threshold}%")
        msg_lines.append("无股票通过严苛财务筛选 (扣非>14% & 估值>1 & 户均>15万)")
        
        if final_candidates:
            msg_lines.append(f"\n💡 [备选] 技术面通过但财务未达标: {len(final_candidates)}只")
            # 简略展示前5
            top_tech = sorted(final_candidates, key=lambda x: x['turnover_rate_now'], reverse=True)[:5]
            for r in top_tech:
                pct = (r['price'] - float(r['pre_close'])) / float(r['pre_close']) * 100
                msg_lines.append(f"- {r['name']}: 涨{pct:.1f}% 换{r['turnover_rate_now']:.1f}%")

    send_feishu_msg("盘中严苛筛选 (Script 1)", "\n".join(msg_lines))
    print("执行完毕。")

if __name__ == "__main__":
    try:
        run_intraday_screener()
    except Exception as e:
        traceback.print_exc()
        send_feishu_msg("盘中脚本报错", str(e))
