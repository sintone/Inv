# -*- coding: utf-8 -*-
"""
@File: tushare_etf_screener_pro_feishu_strict_full_display.py
@Author: Gemini AI (Strict Port)
@Date: 2025-10-12
@Description:
    A股 ETF 技术筛选器 (Pro版) + 飞书通知
    !!! 严格保持原始逻辑，仅在报告输出环节增加Webhook发送 !!!
    !!! 修改：取消显示数量限制，显示所有符合条件的结果 !!!
"""
import pandas as pd
import datetime
import numpy as np
import time
import talib
import tushare as ts
from tqdm import tqdm
import traceback
import requests  # 必需库：用于发送飞书消息
import json      # 必需库：用于封装消息格式

# ==============================================================================
#                                 配置区域
# ==============================================================================
# 请在此处填写您的Tushare Pro接口TOKEN
TUSHARE_TOKEN = "b249a314d4a8db3e43f44db9d5524f31f3425fde397fc9c4633bf9a9"

# 您的 Webhook 地址
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/79bd76b5-cc41-4e81-a971-0a6eec6c1a14"

# 手动指定日期 (格式 YYYYMMDD)，留空则自动计算最新交易日
MANUAL_TRADE_DATE = "" 
# ==============================================================================

def initialize_tushare():
    """初始化 Tushare Pro API"""
    if not TUSHARE_TOKEN:
        exit("错误：请配置 TUSHARE_TOKEN。")
    return ts.pro_api(TUSHARE_TOKEN)

# --- 新增辅助函数：发送飞书消息 ---
def send_feishu_msg(text):
    if not text: return
    headers = {'Content-Type': 'application/json'}
    # 注意：如果文本非常长（超过30k字符），飞书API可能会报错，请留意
    payload = {"msg_type": "text", "content": {"text": text}}
    try:
        requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        print(">> 飞书通知已发送")
    except Exception:
        print(">> 飞书发送失败 (网络错误)")
# -------------------------------

def get_latest_trade_date(pro) -> str:
    """自动获取最新交易日 (完全复制自原始代码)"""
    print("\n步骤 1 - 确定最新交易日...")
    now = datetime.datetime.now()
    today_str = now.strftime('%Y%m%d')

    try:
        # 查询上交所日历
        df_cal = pro.trade_cal(exchange='SSE', cal_date=today_str, fields="cal_date,is_open,pretrade_date")
        if df_cal is None or df_cal.empty:
            return None

        today_info = df_cal.iloc[0]
        
        # 16:00 后且是交易日，则取今日，否则取上一交易日 (此处为您原始代码逻辑)
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
    """构建ETF池 (完全复制自原始代码)"""
    print("\n步骤 2 - 构建ETF品种池...")
    try:
        # 获取单日行情
        daily_df = pro.fund_daily(trade_date=trade_date, fields='ts_code,amount')
        if daily_df is None or daily_df.empty:
            return None

        # 获取基础信息
        basic_info_df = pro.fund_basic(market='E', fields='ts_code,name')
        
        # 合并
        merged_df = pd.merge(daily_df, basic_info_df, on='ts_code', how='inner')
        merged_df.dropna(subset=['amount', 'name'], inplace=True)
        
        # 筛选成交额 > 0 并排序取前200
        etf_pool_df = merged_df[merged_df['amount'] > 0].sort_values(by='amount', ascending=False).head(200)
        etf_pool_df = etf_pool_df[['ts_code', 'name']].rename(columns={'name': 'etf_name'})

        print(f"步骤 2 - 完成。ETF池共 {len(etf_pool_df)} 只。")
        return etf_pool_df
    except Exception as e:
        traceback.print_exc()
        return None

def analyze_etfs(pro, etf_pool: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """计算核心指标 (完全复制自原始代码)"""
    print("\n步骤 3 - 计算均线及各项指标...")
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=250)).strftime('%Y%m%d')
    results_list = []

    for _, row in tqdm(etf_pool.iterrows(), total=etf_pool.shape[0], desc="Processing"):
        ts_code = row['ts_code']
        etf_name = row['etf_name']
        
        # 遵守Tushare流控 (每分钟500次左右)
        time.sleep(0.12)
        
        try:
            # 获取K线
            hist_df = pro.fund_daily(
                ts_code=ts_code, 
                start_date=start_date, 
                end_date=end_date,
                fields='trade_date,close,pct_chg,amount'
            )
            
            if hist_df is None or hist_df.empty or len(hist_df) < 121: continue
            
            # Tushare数据默认是日期降序，需转为升序计算MA
            hist_df.sort_values('trade_date', ascending=True, inplace=True)
            
            # 校验日期
            if hist_df.iloc[-1]['trade_date'] != end_date: continue

            closes = hist_df['close'].values.astype(float)
            if np.isnan(closes).any(): continue

            # --- 计算 MA ---
            ma20 = talib.SMA(closes, 20)
            ma60 = talib.SMA(closes, 60)
            ma120 = talib.SMA(closes, 120)
            
            if np.isnan([ma20[-1], ma60[-1], ma120[-1]]).any(): continue

            # 数据准备
            c_now = closes[-1]
            m20_now, m60_now, m120_now = ma20[-1], ma60[-1], ma120[-1]
            
            c_prev = closes[-2]
            m20_prev, m60_prev, m120_prev = ma20[-2], ma60[-2], ma120[-2]
            
            max_ma_now = max(m20_now, m60_now, m120_now)
            min_ma_now = min(m20_now, m60_now, m120_now)
            max_ma_prev = max(m20_prev, m60_prev, m120_prev)
            min_ma_prev = min(m20_prev, m60_prev, m120_prev)

            # --- 核心筛选条件 (移植逻辑) ---
            
            # 1. 均线交叉
            is_cross_up = (c_now > max_ma_now) and (c_prev <= max_ma_prev)
            is_cross_down = (c_now < min_ma_now) and (c_prev >= min_ma_prev)
            
            # 2. 持续形态
            is_bull_sustain = (c_now > max_ma_now) and (c_prev > max_ma_prev)
            is_bear_sustain = (c_now < min_ma_now) and (c_prev < min_ma_prev)
            
            # 3. 乖离率
            bias_up_val = (c_now - max_ma_now) / max_ma_now
            is_bias_up = bias_up_val > 0.15 
            
            bias_down_val = (min_ma_now - c_now) / min_ma_now
            is_bias_down = bias_down_val > 0.15

            # 4. 7日涨跌幅
            day_7_chg = 0.0
            if len(hist_df) >= 8:
                # Tushare 收盘价未复权，计算涨幅仅供参考
                c_7days_ago = closes[-8]
                day_7_chg = (c_now - c_7days_ago) / c_7days_ago * 100

            # 存储结果
            res = {
                'ts_code': ts_code, 'etf_name': etf_name, 
                'close': c_now, 'pct_chg': float(hist_df.iloc[-1]['pct_chg']), 
                'amount': float(hist_df.iloc[-1]['amount']) / 1000, # 转为万元
                'day_7_chg': day_7_chg,
                'flag_cross_up': is_cross_up,
                'flag_cross_down': is_cross_down,
                'flag_bull_sustain': is_bull_sustain,
                'flag_bear_sustain': is_bear_sustain,
                'flag_bias_up_15': is_bias_up,
                'flag_bias_down_15': is_bias_down,
            }
            results_list.append(res)
        except Exception:
            pass
            
    if not results_list:
        return pd.DataFrame()
    
    return pd.DataFrame(results_list)

def print_simple_line(row):
    """(修改) 不再直接print，而是返回字符串供收集"""
    return f"  代码: {row['etf_name']} ({row['ts_code']}), 收盘价: {row['close']:.3f}, 涨跌幅: {row['pct_chg']}%, 7日涨跌幅: {row['day_7_chg']:.2f}%, 成交额: {row['amount']:.0f}万元"

def generate_report(master_df: pd.DataFrame, trade_date: str):
    """(修改) 生成报告并发送"""
    
    msg_lines = [] # 用于收集所有文字
    
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
    
    # 定义输出组
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
            # 排序规则：7日涨幅
            # [修改点] 移除 .head(10) 和 尾部截断逻辑，全部显示
            for _, row in df_sub.sort_values('day_7_chg', ascending=False).iterrows():
                log(print_simple_line(row))

    # 7日排行 (排行榜通常保留Top10更有意义，如果您也希望这里全部显示，请告诉我，下面依然保留前10)
    top10_gainers = master_df.sort_values('day_7_chg', ascending=False).head(10)
    top10_losers = master_df.sort_values('day_7_chg', ascending=True).head(10)

    log(f"\n=== 7日涨幅榜前10 ===")
    for _, row in top10_gainers.iterrows():
        log(print_simple_line(row))

    log(f"\n=== 7日跌幅榜前10 ===")
    for _, row in top10_losers.iterrows():
        log(print_simple_line(row))
        
    # 保存 CSV
    output_file = f'ETF_Simple_Screener_{trade_date}.csv'
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f"\n结果已保存: {output_file}") # 这个不发飞书

    # --- 发送飞书 ---
    send_feishu_msg("\n".join(msg_lines))

def main():
    try:
        pro = initialize_tushare()
        
        # 1. 确定日期
        last_trade_day = MANUAL_TRADE_DATE if MANUAL_TRADE_DATE else get_latest_trade_date(pro)
        if not last_trade_day: return

        # 2. 构建池子
        etf_pool = build_etf_pool(pro, last_trade_day)
        if etf_pool is None: return

        # 3. 分析
        master_df = analyze_etfs(pro, etf_pool, last_trade_day)
        
        # 4. 报告
        generate_report(master_df, last_trade_day)

    except KeyboardInterrupt:
        print("\n程序中断")
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    main()
