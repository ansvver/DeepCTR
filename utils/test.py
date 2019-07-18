import  time

second = 53087134021


local_time = time.localtime(1561884053)
year = local_time.tm_year - 3649
month = local_time.tm_mon - 1
mday = local_time.tm_mday - 1
wday = local_time.tm_wday
hour = local_time.tm_hour
minute = local_time.tm_min
second = local_time.tm_sec
month_day = month * 31 + mday

hour_bin = hour // 6
minute_bin = minute // 6
season = month // 4
y_m_d_h = 6 * (31 * (year * 12 + month) + mday) + hour_bin

print(year)
print(month)
print(mday)