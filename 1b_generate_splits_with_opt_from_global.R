# library(devtools)
# install_github("MaStatLab/PTT")

library(tidyverse)
library(PTT)
library(cowplot)
library(collections)

set.seed(345345)

df = read_csv("./processed_data/rideaustin_productivity.csv")
hist(df$productivity)


# to produce splits we follow the example from
#   https://github.com/MaStatLab/PTT/blob/master/Examples/example_1d.R


# ------------------------------

# NO SQRT ROOT

# sample_size = 85000
# prod_sample = sample(df$productivity, sample_size)

prod_sample = df$productivity
X = matrix(prod_sample, ncol=1)

max_X = max(X)
min_X = min(X)
range_X = max_X - min_X

max.resol = 6
mod = opt(
  X=(X - min_X) / range_X,
  Xpred=seq(0.0, 1.0, length.out=500),
  max.resol=max.resol,
  rho0=0.1,
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

sprintf("Using %s splits", nrow(splits_opt))

splits_df = tibble(
  val=c(splits_qua, splits_opt$mid, splits_uni),
  type=c(
    rep("Quantiles (31 splits)", 31),
    rep("Optional Polya Tree (36 splits)", nrow(splits_opt)),
    rep("Uniform (31 splits)", 31)
  )
)


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
  )
porig

ggsave(
  "processed_data/splits_original_scale.pdf",
  porig,
  width=10,
  height=12,
  units="cm"
)




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
  width=10,
  height=12,
  units="cm"
)

