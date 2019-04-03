import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(30, 3), columns = list('xyz'))
df[(df.x < df.y) & (df.y < df.z)]
df.query('(x < y) & (y < z)')
df.query('index > y > z')

###
df = pd.DataFrame({"a": list('abbcccf'), "b": list("bccaadd")})
df.query('a in b')
df.query('a not in b')
df.query('b == ["a", "b"]')

###
s = pd.Series(range(5))
s[s > 1]
s.where(s > 1)
s.mask(s > 1)

###
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()
s.str.upper()
s.str.len()

###
df = pd.DataFrame(np.random.randn(5, 2), columns = ['First Name', 'Last Name'], index = range(5))
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

###
pd.Series(['a_b_c', 'c_d_e']).str.split('_')
pd.Series(['a_b_c', 'c_d_e']).str.split('_').str.get(0)
pd.Series(['a_b_c', 'c_d_e']).str.split('_', expand = True)
pd.Series(['a_b_c', 'c_d_e']).str.replace('^a', 'xxxx', case = False)

###
pd.to_datetime(['1/1/2019', np.datetime64('2019-01-01')])
pd.date_range('2019-01-01', periods = 3, freq = 'H')
pd.date_range('2019-01-01', periods = 3, freq = 'H').tz_localize('UTC')
pd.date_range('2019-01-01', periods = 3, freq = 'H').tz_localize('UTC').tz_convert("US/Pacific")

###
ts = pd.Series(range(5), index = pd.date_range("2019-01-01", periods = 5, freq = 'H'))
ts.resample("2H").mean()

###
pd.Timestamp('2019-01-04').day_name()
pd.Timestamp('2019-01-04') + pd.offsets.BDay()
pd.Timestamp('2019-01-04') + pd.Timedelta('1 day')

###
ts['1/1/2019 01:00']
import datetime
ts[datetime.datetime(2019, 1, 1, 1)]