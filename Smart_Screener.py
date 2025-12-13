# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import time
import talib
import os
import requests
import json
from tqdm import tqdm
from datetime import datetime, timedelta

# --- 全局配置 ---
# 优先从环境变量读取，本地运行则使用默认值
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", 'b249a314d4a8db3e43f44db9d5524f31f3425fde397fc9c4633bf9a9')
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
API_CALL_DELAY = 0.06  

# --- 缓存 ---
cache_data = {
    'daily_bar': {},
    'holders': {},
    'financial_snapshot': {}
}

# --- 飞书发送模块 (新增) ---
def send_feishu_summary(result_dict):
    """
    将筛选结果整理为文本摘要发送到飞书
    """
    if not FEISHU_WEBHOOK_URL:
        print(">> 未配置 FEISHU_WEBHOOK，跳过发送")
        return

    msg_lines = []
    current_time = datetime.now().strftime('%m-%d %H:%M')
    msg_lines.append(f"📊 **精算版筛选报告** ({current_time})")
    msg_lines.append("----------------")

    has_data = False
    
    # 遍历结果字典
    for title, df in result_dict.items():
        if df.empty:
            continue
        
        has_data = True
        # 简化标题，去掉前面的数字编号
        clean_title = title.split('_')[-1] if '_' in title else title
        msg_lines.append(f"\n📌 {clean_title} (Top 5/{len(df)})")
        
        # 只取前5个展示在飞书，避免消息过长
        top_n = df.head(5)
        for _, row in top_n.iterrows():
            # 格式：名称(代码) | 估值:X | 7日:Y%
            name = row.get('name', 'N/A')
            code = row.get('ts_code', '')
            val_ratio = row.get('valuation_ratio', 0)
            chg7 = row.get('day_7_chg', 0)
            
            # 处理可能的 NaN
            val_str = f"{val_ratio:.2f}" if pd.notna(val_ratio) else "-"
            chg_str = f"{chg7:.1f}%" if pd.notna(chg7) else "-"
            
            line = f"{name}({code}) | 估值:{val_str} | 7日:{chg_str}"
            msg_lines.append(line)

    if not has_data:
        msg_lines.append("今日无符合条件的筛选结果。")
    else:
        msg_lines.append("\n💡 完整HTML报告已保存至 GitHub Artifacts")

    # 发送请求
    full_text = "\n".join(msg_lines)
    headers = {'Content-Type': 'application/json'}
    payload = {"msg_type": "text", "content": {"text": full_text}}
    
    try:
        requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        print(">> 飞书摘要已发送")
    except Exception as e:
        print(f">> 飞书发送失败: {e}")

# --- 基础工具 (保持原逻辑) ---

def get_last_trade_day():
    now = datetime.now()
    if now.hour < 16: now = now - timedelta(days=1)
    for _ in range(15):
        date_str = now.strftime('%Y%m%d')
        try:
            df = pro.trade_cal(exchange='', start_date=date_str, end_date=date_str)
            if not df.empty and df.iloc[0]['is_open'] == 1:
                check = pro.daily(trade_date=date_str, limit=1)
                if not check.empty:
                    print(f"锁定最近交易日: {date_str}")
                    return date_str
        except: pass
        now = now - timedelta(days=1)
    return datetime.now().strftime('%Y%m%d')

def get_stock_price_data(ts_code, end_date, n_days=200):
    if ts_code in cache_data['daily_bar']: return cache_data['daily_bar'][ts_code]
    start_date = (pd.to_datetime(end_date) - timedelta(days=n_days*2)).strftime('%Y%m%d')
    try:
        time.sleep(API_CALL_DELAY)
        df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date)
        if df is None or df.empty: return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 120: return None
        close = df['close'].values
        df['ma20'] = talib.SMA(close, timeperiod=20)
        df['ma60'] = talib.SMA(close, timeperiod=60)
        df['ma120'] = talib.SMA(close, timeperiod=120)
        cache_data['daily_bar'][ts_code] = df
        return df
    except: return None

def get_latest_financial_snapshot(ts_code, trade_date):
    if ts_code in cache_data['financial_snapshot']: return cache_data['financial_snapshot'][ts_code]
    res = {'revenue': np.nan, 'profit_dedt': np.nan, 'debt_to_assets': np.nan, 'acct_recv': 0.0, 'total_assets': np.nan, 'ocf': np.nan, 'total_mv_wan': np.nan}
    start_dt = (pd.to_datetime(trade_date) - timedelta(days=450)).strftime('%Y%m%d')
    try:
        time.sleep(API_CALL_DELAY)
        df_mv = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields='total_mv')
        if not df_mv.empty: res['total_mv_wan'] = df_mv.iloc[0]['total_mv']
        
        # 简化合并逻辑，保留核心获取部分
        time.sleep(API_CALL_DELAY)
        df_bal = pro.balancesheet(ts_code=ts_code, start_date=start_dt, fields='end_date,acct_recv,accounts_receiv,total_assets')
        if not df_bal.empty:
            latest = df_bal.sort_values('end_date', ascending=False).iloc[0]
            ar = latest.get('acct_recv') if pd.notna(latest.get('acct_recv')) else latest.get('accounts_receiv')
            res['acct_recv'] = ar if pd.notna(ar) else 0.0
            res['total_assets'] = latest['total_assets']

        time.sleep(API_CALL_DELAY)
        df_cash = pro.cashflow(ts_code=ts_code, start_date=start_dt, fields='end_date,n_cashflow_act')
        if not df_cash.empty: res['ocf'] = df_cash.sort_values('end_date', ascending=False).iloc[0]['n_cashflow_act']

        time.sleep(API_CALL_DELAY)
        df_inc = pro.income(ts_code=ts_code, start_date=start_dt, fields='end_date,revenue')
        if not df_inc.empty: res['revenue'] = df_inc.sort_values('end_date', ascending=False).iloc[0]['revenue']

        time.sleep(API_CALL_DELAY)
        df_ind = pro.fina_indicator(ts_code=ts_code, start_date=start_dt, fields='end_date,profit_dedt,debt_to_assets')
        if not df_ind.empty:
            latest_ind = df_ind.sort_values('end_date', ascending=False).iloc[0]
            res['profit_dedt'] = latest_ind['profit_dedt']
            res['debt_to_assets'] = latest_ind['debt_to_assets']
    except: pass
    cache_data['financial_snapshot'][ts_code] = res
    return res

def get_precise_per_capita_mv(ts_code, total_float_share, close_price, trade_date):
    try:
        start_dt = (pd.to_datetime(trade_date) - timedelta(days=365)).strftime('%Y%m%d')
        if ts_code in cache_data['holders']:
            holder_num = cache_data['holders'][ts_code]
        else:
            time.sleep(API_CALL_DELAY)
            df_h = pro.stk_holdernumber(ts_code=ts_code, start_date=start_dt)
            if df_h.empty: return None
            holder_num = df_h.sort_values('end_date', ascending=False).iloc[0]['holder_num']
            cache_data['holders'][ts_code] = holder_num
        if holder_num <= 10: return None 

        time.sleep(API_CALL_DELAY) 
        df_top10 = pro.top10_floatholders(ts_code=ts_code, start_date=start_dt)
        top10_shares = 0.0
        if not df_top10.empty:
            latest_date = df_top10['end_date'].max()
            top10_shares = df_top10[df_top10['end_date'] == latest_date]['hold_amount'].sum()
        
        retail_shares = (total_float_share * 10000) - top10_shares
        if retail_shares < 0: retail_shares = 0
        retail_holders = holder_num - 10
        if retail_holders <= 0: return None
        return (retail_shares * close_price) / retail_holders
    except: return None

# --- 核心逻辑 ---

class SmartSelector:
    def __init__(self):
        self.trade_date = get_last_trade_day()
        self.pool = pd.DataFrame() 
        self.output_columns_map = {
            'ts_code': 'ts_code', 'name': 'name', 'per_capita_mv_wan': '户均流通市值(万)',
            'industry': 'industry', 'concept_name': 'concept', 'pct_chg': '当日涨跌幅',
            'day_7_chg': '7日涨跌幅(%)', 'turnover_rate': '换手率', 'valuation_ratio': 'valuation_ratio',
            'ar_ratio': '应收账款占比(%)', 'debt_to_assets': '负债率(%)', 'ocf_val': '经营现金流(万)'
        }

    def step1_global_filter(self):
        print(f"\n>>> 步骤1: 构建精品池 (交易日: {self.trade_date})")
        df_daily = pro.daily_basic(trade_date=self.trade_date, fields='ts_code,close,turnover_rate,float_share,total_mv')
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        df_merge = pd.merge(df_daily, df_basic, on='ts_code')
        valid_stocks = []
        for index, row in tqdm(df_merge.iterrows(), total=len(df_merge), desc="计算户均市值"):
            try:
                if pd.isna(row['close']) or pd.isna(row['float_share']): continue
                res_val = get_precise_per_capita_mv(row['ts_code'], row['float_share'], row['close'], self.trade_date)
                if res_val and res_val > 150000:
                    row['per_capita_mv_wan'] = round(res_val / 10000, 2)
                    if pd.notna(row['turnover_rate']): row['turnover_rate'] = round(row['turnover_rate'], 2)
                    valid_stocks.append(row)
            except: continue
        self.pool = pd.DataFrame(valid_stocks)
        print(f"\n步骤1完成: 筛选出 {len(self.pool)} 只。")
        if not self.pool.empty:
            tqdm.pandas(desc="补充概念")
            def get_concept(code):
                try:
                    df = pro.concept_detail(ts_code=code)
                    if not df.empty: return df.iloc[0]['concept_name']
                except: pass
                return ''
            self.pool['concept_name'] = self.pool['ts_code'].progress_apply(get_concept)

    def step2_calc_features(self):
        if self.pool.empty: return
        print("\n>>> 步骤2: 计算指标...")
        cols = ['ma20', 'ma60', 'ma120', 'close_prev', 'ma20_prev', 'ma60_prev', 'ma120_prev', 'valuation_ratio', 'ar_ratio', 'debt_to_assets', 'ocf_val', 'day_7_chg', 'bias_up_pct', 'bias_down_pct']
        for col in cols: self.pool[col] = np.nan
        
        for idx, row in tqdm(self.pool.iterrows(), total=len(self.pool), desc="综合计算"):
            ts_code = row['ts_code']
            fin = get_latest_financial_snapshot(ts_code, self.trade_date)
            tmv_wan = fin['total_mv_wan'] if pd.notna(fin['total_mv_wan']) else row['total_mv']
            if pd.notna(fin['revenue']) and fin['revenue']!=0 and pd.notna(fin['profit_dedt']) and pd.notna(tmv_wan) and tmv_wan>0:
                self.pool.at[idx, 'valuation_ratio'] = round((fin['revenue'] * (fin['profit_dedt']/fin['revenue'] / 0.14) * 10) / (tmv_wan * 10000), 2)
            
            ar, ast = fin['acct_recv'], fin['total_assets']
            self.pool.at[idx, 'ar_ratio'] = round((ar/ast)*100, 2) if pd.notna(ast) and ast!=0 else 0.0
            self.pool.at[idx, 'debt_to_assets'] = round(fin['debt_to_assets'], 2) if pd.notna(fin['debt_to_assets']) else 0.0
            self.pool.at[idx, 'ocf_val'] = int(fin['ocf']/10000) if pd.notna(fin['ocf']) else np.nan

            df_bar = get_stock_price_data(ts_code, self.trade_date)
            if df_bar is not None:
                curr, prev = df_bar.iloc[-1], df_bar.iloc[-2]
                self.pool.at[idx, 'pct_chg'] = curr['pct_chg']
                for ma in ['ma20','ma60','ma120']: self.pool.at[idx, ma] = curr[ma]; self.pool.at[idx, f"{ma}_prev"] = prev[ma]
                self.pool.at[idx, 'close_prev'] = prev['close']
                if len(df_bar) >= 8: self.pool.at[idx, 'day_7_chg'] = round((curr['close'] - df_bar.iloc[-8]['close'])/df_bar.iloc[-8]['close'] * 100, 2)
                
                c = curr['close']
                ma_vals = [curr['ma20'], curr['ma60'], curr['ma120']]
                if all(pd.notna(x) for x in ma_vals):
                    self.pool.at[idx, 'bias_up_pct'] = (c - max(ma_vals))/max(ma_vals) if c > max(ma_vals) else np.nan
                    self.pool.at[idx, 'bias_down_pct'] = (min(ma_vals) - c)/min(ma_vals) if c < min(ma_vals) else np.nan

    def step3_generate_sublists(self):
        print("\n>>> 步骤3: 生成列表...")
        df = self.pool.dropna(subset=['ma120']).copy()
        res = {}
        
        def check_cross_up(r): return r['close']>max(r['ma20'],r['ma60'],r['ma120']) and r['close_prev']<=max(r['ma20_prev'],r['ma60_prev'],r['ma120_prev'])
        def check_cross_down(r): return r['close']<min(r['ma20'],r['ma60'],r['ma120']) and r['close_prev']>=min(r['ma20_prev'],r['ma60_prev'],r['ma120_prev'])
        def check_bull(r): return r['close']>max(r['ma20'],r['ma60'],r['ma120']) and r['close_prev']>max(r['ma20_prev'],r['ma60_prev'],r['ma120_prev'])
        def check_bear(r): return r['close']<min(r['ma20'],r['ma60'],r['ma120']) and r['close_prev']<min(r['ma20_prev'],r['ma60_prev'],r['ma120_prev'])

        res['2_收盘价上穿三条均线'] = df[df.apply(check_cross_up, axis=1)].sort_values('valuation_ratio', ascending=False)
        res['3_收盘价下穿三条均线'] = df[df.apply(check_cross_down, axis=1)].sort_values('valuation_ratio', ascending=False)
        res['4_上涨乖离(>15%)'] = df[df['bias_up_pct']>0.15].sort_values('valuation_ratio', ascending=False)
        res['5_下跌乖离(>15%)'] = df[df['bias_down_pct']>0.15].sort_values('valuation_ratio', ascending=False)
        res['6_7天涨幅前30'] = df.sort_values('day_7_chg', ascending=False).head(30).sort_values('valuation_ratio', ascending=False)
        res['7_7天跌幅前30'] = df.sort_values('day_7_chg', ascending=True).head(30).sort_values('valuation_ratio', ascending=False)
        res['8_持续多头形态'] = df[df.apply(check_bull, axis=1)].sort_values('valuation_ratio', ascending=False)
        res['9_持续空头形态'] = df[df.apply(check_bear, axis=1)].sort_values('valuation_ratio', ascending=False)
        return res

    def generate_html_report(self, result_dict):
        # 保持原有的HTML生成逻辑不变，只修改文件写入路径适应GitHub Actions
        print("\n>>> 生成HTML报表...")
        html_content = f"<html><head><meta charset='utf-8'><title>筛选报告</title></head><body><h1>精算版筛选报告 {datetime.now()}</h1>"
        for title, df in result_dict.items():
            html_content += f"<h2>{title} ({len(df)})</h2>"
            if not df.empty:
                final_cols = [c for c in self.output_columns_map.keys() if c in df.columns]
                html_content += df[final_cols].rename(columns=self.output_columns_map).to_html(index=False, classes='display')
        html_content += "</body></html>"
        
        # 固定文件名，方便GitHub Artifacts上传
        filename = "smart_report.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML文件已保存: {filename}")

if __name__ == "__main__":
    app = SmartSelector()
    app.step1_global_filter()
    app.step2_calc_features()
    results = app.step3_generate_sublists()
    
    # 1. 生成 HTML (供 Artifacts 上传)
    app.generate_html_report(results)
    
    # 2. 发送飞书文本摘要 (直接推送到手机)
    send_feishu_summary(results)
