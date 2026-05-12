import yfinance as yf

symbol = "1810.HK"

df = yf.download(
    symbol,
    start="2021-01-01",
    end="2026-05-12"
)

print(df.head())
print(df.tail())
print(df.shape)

df.to_csv("xiaomi_1810_HK.csv")

print("已保存为 xiaomi_1810_HK.csv")