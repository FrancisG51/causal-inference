library(mice)
library(here)

setwd(here('data'))

all_R_mice = read.csv('data_to_mice.csv')
all_R_mice = mice(all_R_mice, m=5, seed=123, method='rf')
all_R_mice1 = complete(all_R_mice, action=1)
write.csv(all_R_mice1, 'data_miced.csv', row.names=FALSE)


