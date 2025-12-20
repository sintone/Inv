# -*- coding: utf-8 -*-
"""
@File: Daily_ETF.py
@Description: A股 ETF 技术筛选器 (Pro版) - 已适配 GitHub Secrets 环境
"""
import pandas as pd
import datetime
import numpy as np
import time
import talib
import tushare as ts
from tqdm import tqdm
import traceback
import requests
import json
import os  # 新增：用于读取环境变量

# ==============================================================================
#                                配置区域 (已修改)
# ==============================================================================
# 从环境变量中读取敏感信息，如果读取不到则为空字符串
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

# 手动指定日期 (格式 YYYYMMDD)，留空则自动计算最新交易日
MANUAL_TRADE_DATE = "" 
# ==============================================================================

def initialize_tushare():
    """初始化 Tushare Pro API"""
    if not TUSHARE_TOKEN:
        exit("错误：未在环境变量中找到 TUSHARE_TOKEN。请检查 GitHub Secrets 配置。")
    return ts.pro_api(TUSHARE_TOKEN)

# --- 辅助函数：发送飞书消息 ---
def send_feishu_msg(text):
    if not text: return
    if not FEISHU_WEBHOOK_URL:
        print(">> 未配置飞书 Webhook，跳过消息发送。")
        return
    headers = {'Content-Type': 'application/json'}
    payload = {"msg_type": "text", "content": {"text": text}}
    try:
        resp = requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        if resp.status_code == 200:
            print(">> 飞书通知已发送")
        else:
            print(f">> 飞书发送失败: {resp.text}")
    except Exception:
        print(">> 飞书发送失败 (网络错误)")

# ------------------------------------------------------------------------------
# 以下部分核心逻辑完全保持原样，未作改动
# ------------------------------------------------------------------------------

def get_latest_trade_date(pro) -> str:
    print("\n步骤 1 - 确定最新交易日...")
    now = datetime.datetime.now()
    today_str = now.strftime('%Y%m%d')
    try:
        df_cal = pro.trade_cal(exchange='SSE', cal_date=today_str, fields="cal_date,is_open,pretrade_date")
        if df_cal is None or df_cal.empty: return None
        today_info = df_cal.iloc[0]
        if today_info['is_open'] == 1 and now.hour >= 16:
            latest_day = today_info['cal_date']
        else:
            latest_day = today_info['pretrade_date']
        print(f"步骤 1 - 完成。最新交易日为: {pd.to_datetime(latest_day).strftime('%Y-%m-%d')}")
        return latest_day
    except Exception as e:
        print(f"错误：获取日期失败: {e}")
        return None

def build_etf_pool(pro, trade_date: str) -> pd.DataFrame:
    print("\n步骤 2 - 构建ETF品种池...")
    try:
        daily_df = pro.fund_daily(trade_date=trade_date, fields='ts_code,amount')
        if daily_df is None or daily_df.empty: return None
        basic_info_df = pro.fund_basic(market='E', fields='ts_code,name')
        merged_df = pd.merge(daily_df, basic_info_df, on='ts_code', how='inner')
        merged_df.dropna(subset=['amount', 'name'], inplace=True)
        etf_pool_df = merged_df[merged_df['amount'] > 0].sort_values(by='amount', ascending=False).head(200)
        etf_pool_df = etf_pool_df[['ts_code', 'name']].rename(columns={'name': 'etf_name'})
        print(f"步骤 2 - 完成。ETF池共 {len(etf_pool_df)} 只。")
        return etf_pool_df
    except Exception as e:
        traceback.print_exc()
        return None

def analyze_etfs(pro, etf_pool: pd.DataFrame, end_date: str) -> pd.DataFrame:
    print("\n步骤 3 - 计算均线及各项指标...")
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=250)).strftime('%Y%m%d')
    results_list = []
    for _, row in tqdm(etf_pool.iterrows(), total=etf_pool.shape[0], desc="Processing"):
        ts_code = row['ts_code']
        etf_name = row['etf_name']
        time.sleep(0.12) # 流控
        try:
            hist_df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='trade_date,close,pct_chg,amount')
            if hist_df is None or hist_df.empty or len(hist_df) < 121: continue
            hist_df.sort_values('trade_date', ascending=True, inplace=True)
            if hist_df.iloc[-1]['trade_date'] != end_date: continue
            closes = hist_df['close'].values.astype(float)
            if np.isnan(closes).any(): continue
            ma20 = talib.SMA(closes, 20)
            ma60 = talib.SMA(closes, 60)
            ma120 = talib.SMA(closes, 120)
            if np.isnan([ma20[-1], ma60[-1], ma120[-1]]).any(): continue
            c_now, c_prev = closes[-1], closes[-2]
            m20_now, m60_now, m120_now = ma20[-1], ma60[-1], ma120[-1]
            m20_prev, m60_prev, m120_prev = ma20[-2], ma60[-2], ma120[-2]
            max_ma_now, min_ma_now = max(m20_now, m60_now, m120_now), min(m20_now, m60_now, m120_now)
            max_ma_prev, min_ma_prev = max(m20_prev, m60_prev, m120_prev), min(m20_prev, m60_prev, m120_prev)
            is_cross_up = (c_now > max_ma_now) and (c_prev <= max_ma_prev)
            is_cross_down = (c_now < min_ma_now) and (c_prev >= min_ma_prev)
            is_bull_sustain = (c_now > max_ma_now) and (c_prev > max_ma_prev)
            is_bear_sustain = (c_now < min_ma_now) and (c_prev < min_ma_prev)
            bias_up_val = (c_now - max_ma_now) / max_ma_now
            is_bias_up = bias_up_val > 0.15 
            bias_down_val = (min_ma_now - c_now) / min_ma_now
            is_bias_down = bias_down_val > 0.15
            day_7_chg = 0.0
            if len(hist_df) >= 8:
                c_7days_ago = closes[-8]
                day_7_chg = (c_now - c_7days_ago) / c_7days_ago * 100
            res = {
                'ts_code': ts_code, 'etf_name': etf_name, 
                'close': c_now, 'pct_chg': float(hist_df.iloc[-1]['pct_chg']), 
                'amount': float(hist_df.iloc[-1]['amount']) / 1000, 
                'day_7_chg': day_7_chg,
                'flag_cross_up': is_cross_up,
                'flag_cross_down': is_cross_down,
                'flag_bull_sustain': is_bull_sustain,
                'flag_bear_sustain': is_bear_sustain,
                'flag_bias_up_15': is_bias_up,
                'flag_bias_down_15': is_bias_down,
            }
            results_list.append(res)
        except Exception: pass
    if not results_list: return pd.DataFrame()
    return pd.DataFrame(results_list)

def print_simple_line(row):
    return f"  代码: {row['etf_name']} ({row['ts_code']}), 收盘价: {row['close']:.3f}, 涨跌幅: {row['pct_chg']}%, 7日涨跌幅: {row['day_7_chg']:.2f}%, 成交额: {row['amount']:.0f}万元"

def generate_report(master_df: pd.DataFrame, trade_date: str):
    msg_lines = [] 
    def log(text):
        print(text)
        msg_lines.append(text)
    if master_df.empty:
        log("\n结果为空。")
        send_feishu_msg("\n".join(msg_lines))
        return
    log("\n" + "="*60)
    log(f"A股 ETF 筛选结果 (日期: {trade_date})")
    log("="*60)
    categories = [
        ("收盘价上穿三条均线", master_df['flag_cross_up']),
        ("收盘价下穿三条均线", master_df['flag_cross_down']),
        ("上涨乖离率 > 15%", master_df['flag_bias_up_15']),
        ("下跌乖离率 > 15%", master_df['flag_bias_down_15']),
        ("持续多头形态", master_df['flag_bull_sustain']),
        ("持续空头形态", master_df['flag_bear_sustain']),
    ]
    for title, mask in categories:
        df_sub = master_df[mask]
        log(f"\n=== {title} (共 {len(df_sub)} 只) ===")
        if df_sub.empty:
            log("  (无符合条件品种)")
        else:
            for _, row in df_sub.sort_values('day_7_chg', ascending=False).iterrows():
                log(print_simple_line(row))
    top10_gainers = master_df.sort_values('day_7_chg', ascending=False).head(10)
    top10_losers = master_df.sort_values('day_7_chg', ascending=True).head(10)
    log(f"\n=== 7日涨幅榜前10 ===")
    for _, row in top10_gainers.iterrows(): log(print_simple_line(row))
    log(f"\n=== 7日跌幅榜前10 ===")
    for _, row in top10_losers.iterrows(): log(print_simple_line(row))
    output_file = f'ETF_Simple_Screener_{trade_date}.csv'
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.2f')
    send_feishu_msg("\n".join(msg_lines))

def main():
    try:
        pro = initialize_tushare()
        last_trade_day = MANUAL_TRADE_DATE if MANUAL_TRADE_DATE else get_latest_trade_date(pro)
        if not last_trade_day: return
        etf_pool = build_etf_pool(pro, last_trade_day)
        if etf_pool is None: return
        master_df = analyze_etfs(pro, etf_pool, last_trade_day)
        generate_report(master_df, last_trade_day)
    except KeyboardInterrupt: print("\n程序中断")
    except Exception as e: traceback.print_exc()

if __name__ == '__main__':
    main()
