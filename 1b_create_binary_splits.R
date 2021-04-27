# library(devtools)
# install_github("MaStatLab/PTT")

library(tidyverse)
library(PTT)
library(cowplot)
library(collections)

set.seed(123120)

# df = read_csv("./processed_data/rideaustin_productivity.csv")
# hist(df$productivity)


# to produce splits we follow the example from
#   https://github.com/MaStatLab/PTT/blob/master/Examples/example_1d.R


# ------------------------------

# NO SQRT ROOT
prod_sample = df$productivity 

# sample_size = 100000
# prod_sample = sample(prod_sample, sample_size)

X = matrix(prod_sample, ncol=1)

max_X = max(X)
min_X = min(X)
range_X = max_X - min_X

max.resol = 6
mod = opt(
  X=(X - min_X) / range_X,
  Xpred=seq(0.0, 1.0, length.out=500),
  max.resol=max.resol,
  rho0=0.04,
  rho0.mode=0
)

fhat = mod$predictive_densities
fhat = fhat / sum(fhat)
x = min_X + range_X * seq(0.0, 1.0, length.out=500)
plot(x, fhat)


splits_qua = quantile(
  df$productivity,
  seq(2^(-5), 1 - 2^(-5), length.out=2^5 - 1)
)


# add one on left and three on right

min_val = min_X
max_val = 100
num_new_left = 1
num_new_right = 4
new_left = seq(min_val, min(splits_qua), length.out=3)[c(-1, -(num_new_left + 2))]
new_right = seq(max(splits_qua), max_val, length.out=6)[c(-1, -(num_new_right + 2))]
splits_qua_extra = c(
  new_left,
  splits_qua,
  new_right
)
splits_qua_extra

N = 2^5 - 1
splits_uni = max_X * seq(2^(-5), 1 - 2^(-5), length.out=2^5 - 1)

parts = mod$part_points_hmap
d = max.resol
tmp = min_X + range_X * seq(0, 1, length.out=2^d + 1)


splits = c()
lvls = c()
lows = c()
ups = c()

count = 0
set = dict()
for (i in 1:nrow(parts))
  set$set(as.integer(parts[i, 1:2]), 0)

for (i in 1:nrow(parts)) {
  low = parts[i, 1] + 1
  high = parts[i, 2] + 2
  ix_mid = as.integer((parts[i, 2] + parts[i, 1] - 1) %/% 2)
  child_split = c(as.integer(parts[i, 1]), ix_mid)
  if (!is.infinite(parts[i, 4])) {
    # only if not leaf node
    count = count + 1
    lvls[count] = parts[i, 3]
    lows[count] = tmp[low]
    ups[count] = tmp[high] + 1e-12
    splits[count] = 0.5 * (lows[count] + ups[count])
  }
}

splits_opt = tibble(
  lows = lows,
  mid = splits,
  ups = ups,
  lvls = lvls
) %>% 
  arrange(lvls, lows)  # bfs order
write_csv(splits_opt, "processed_data/splits_opt_pt.csv")

# first up to level 5
lows = c()
mids = c()
ups = c()
lvls = c()
min_val = min_X
max_val = 100
tmp = c(min_val, splits_qua, max_val)
c = 1
for (lev in 1:5) {
  delta = 2^(5 - lev)
  num_interv = 2^(lev - 1)
  for (j in 1:num_interv) {
    base = 2 * (j - 1) * delta + 1
    lows[c] = tmp[base]
    mids[c] = tmp[base + delta]
    ups[c] = tmp[base + 2 * delta]
    lvls[c] = lev - 1
    c = c + 1
  }
}
# add manual splits
# left strategy is split new right versus all to the left
tmp = c(new_left, min(splits_qua))
for (k in 1:length(new_left)) {
  lows[c] = min_val
  mids[c] = tmp[length(tmp) - k]
  ups[c] = tmp[length(tmp) - k + 1]
  lvls[c] = 4 + k
  c = c + 1
}
# right strategy is split new left versus all to the right
tmp = c(max(splits_qua), new_right)
for (k in 1:length(new_right)) {
  lows[c] = tmp[k]
  mids[c] = tmp[k + 1]
  ups[c] = max_val
  lvls[c] = 4 + k
  c = c + 1
}

splits_qua_mat = tibble(
  lows = lows,
  mid = mids,
  ups = ups,
  lvls = lvls
) %>% 
  arrange(lvls, lows)  # bfs order
write_csv(splits_qua_mat, "processed_data/splits_qua.csv")




sprintf("Using %s splits", nrow(splits_opt))

splits_df = tibble(
  val=c(splits_qua, splits_qua_extra, splits_opt$mid, splits_uni),
  type=c(
    rep("(C) Quantiles (31 splits)", 31),
    rep("(D) Quantiles Extended (36 splits)", 36),
    rep(sprintf("(B) OptionalPolyaTree(6, 0.1) (%s splits)",  nrow(splits_opt)), nrow(splits_opt)),
    rep("(A) Uniform (31 splits)", 31)
  )
)
# 
# 
porig = ggplot() +
  geom_histogram(
    aes(x=productivity),
    bins=45,
    data=df
  ) +
  geom_vline(aes(xintercept=val, color=type), data=splits_df) +
  facet_wrap(~ type, ncol=1) +
  theme_minimal_hgrid() +
  # labs(
    # title="original scale productivity",
    # subtitle="Optional polya tree splits from original scale"
  # ) +
  # scale_y_log10() +
  labs(y="count", x="productivity ($/h)") +
  guides(color=FALSE) +
  scale_color_brewer(palette="Set2") +
  theme(
    axis.text.x = element_text(size=10),
    axis.text.y = element_text(size=10),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size=10)
  ) + 
  xlim(0, 122)
porig
# 
# 
# 
ggsave(
  "processed_data/splits_original_scale.pdf",
  porig,
  width=12,
  height=12,
  units="cm"
)
# 
# 
# 
# 
plog = ggplot() +
  geom_histogram(
    aes(x=productivity),
    bins=30,
    data=df
  ) +
  geom_vline(aes(xintercept=val, color=type), data=splits_df) +
  facet_wrap(~ type, ncol=1) +
  theme_minimal_hgrid() +
  # labs(
  # title="original scale productivity",
  # subtitle="Optional polya tree splits from original scale"
  # ) +
  scale_y_log10() +
  labs(y="count (log10 scale)", x="productivity ($/h)") +
  guides(color=FALSE) +
  scale_color_brewer(palette="Set2") +
  theme(
    axis.text.x = element_text(size=10),
    axis.text.y = element_text(size=10),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size=10)
  )
plog

ggsave(
  "processed_data/splits_log_scale.pdf",
  plog,
  width=12,
  height=12,
  units="cm"
)

