import polars as pl
from entiny import entiny

a = pl.int_range(1, 30, eager=True)
df = pl.DataFrame({"a": a})

b = df.select(pl.col("a").shuffle(seed=1))
c = df.select(pl.col("a").shuffle(seed=2))

# Add column using the Series, not the DataFrame
df = df.with_columns(
    b=b.to_series(),
    c=c.to_series()
)

print(df)

e = entiny(df, n=1).collect()

print(e)