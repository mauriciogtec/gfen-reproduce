library(tidyverse) 
library(lubridate)
library(raster)
library(igraph)
library(progress)


# read raw data
# raw data can be downloaded from https://data.world/ride-austin (data.world 2017)
# data included here downloaded April 7 2018
data_A <- read_csv("./raw_data/Rides_DataA.csv", locale = locale(tz = "US/Central"))
data_B <- read_csv("./raw_data/Rides_DataB.csv", locale = locale(tz = "US/Central"))
data <-  full_join(data_A, data_B) %>% 
  dplyr::select(RIDE_ID, driver_id,
                started_on, created_date, completed_on,
                requested_car_category, surge_factor,
                start_location_long, start_location_lat,
                end_location_long, end_location_lat,
                base_fare,  distance_traveled)

# read shapefile
taz <- shapefile("./raw_data/shapefiles/TAZs.shp")
taz@data$ID <- as.integer(taz@data$ID)

# tag the TAZ where each ride falls
start_locs = data[ ,c("start_location_long", "start_location_lat")]
start <- SpatialPoints(start_locs, proj4string = taz@proj4string)
end_locs = data[ ,c("end_location_long", "end_location_lat")]
end <- SpatialPoints(end_locs, proj4string = taz@proj4string)
start_taz <- over(start, taz)
end_taz <- over(end, taz)
data$start_taz <- start_taz$ID
data$end_taz <- end_taz$ID

# remove points not in any TAZ
data <- data %>% 
  filter(!is.na(start_taz) & !is.na(end_taz))

# filter unreasonable values
data <- data %>% 
  mutate(duration = as.numeric(difftime(completed_on, started_on, units = "mins"))) %>% 
  rename(distance_meters = distance_traveled) %>% 
  filter(between(distance_meters, 10, 100000)) %>%  # max 100km trip
  filter(between(duration, 1, 120))  # max 2 hour trip

# reestimate fare to remove surge and standardize to regular car
data <- data %>% 
  mutate(weekday_started = weekdays(started_on)) %>% 
  mutate(weekday_completed = weekdays(completed_on)) %>%
mutate(hour_started = lubridate::hour(started_on)) %>% 
  mutate(hour_completed = lubridate::hour(completed_on))

data$surge_factor[is.na(data$surge_factor) | data$surge_factor %in% c(0, 1)] <- 1

data <- data %>% 
  mutate(distance_km = distance_meters / 1000) %>% 
  mutate(distance_miles = distance_km * 0.62137119) %>% 
  mutate(mile_fare_estim = distance_miles * 0.99) %>%
  mutate(time_fare_estim = duration * 0.25) %>% 
  mutate(fare_pre_estimate = base_fare + time_fare_estim + mile_fare_estim) %>% 
  mutate(fare_estimate = pmax(fare_pre_estimate, 4)) %>% 
  mutate(fare_pre_estimate_with_surge = fare_pre_estimate * surge_factor)  %>% 
  mutate(fare_estimate_with_surge = pmax(fare_pre_estimate_with_surge, 4))

# estimate productiity by splitting set for each driver and computing 
#   prospective measurements
print("generating metrics per driver...")
drivers <- unique(data$driver_id)
pb <- progress::progress_bar$new(total = length(drivers))
by_driver <- map(drivers, function(id) {
  pb$tick()
  data %>% 
    filter(driver_id == id) %>%
    arrange(started_on) %>% 
    mutate(idle_after = as.numeric(difftime(lead(created_date), completed_on, units = "mins"))) %>%
    mutate(unproductive_after = as.numeric(difftime(lead(started_on), completed_on, units = "mins"))) %>%
    mutate(reach_after = as.numeric(difftime(lead(started_on), lead(created_date), units = "mins"))) %>%
    mutate(duration_next = as.numeric(difftime(lead(completed_on), lead(started_on), units = "mins"))) %>%
    mutate(fare_next = lead(fare_estimate)) %>%
    mutate(fare_next_with_surge = lead(fare_estimate_with_surge)) %>% 
    mutate(productivity = fare_next / (unproductive_after + duration_next) * 60) %>% 
    mutate(productivity_with_surge = fare_next_with_surge /
             (unproductive_after + duration_next) * 60)
})
data <- bind_rows(by_driver) 
err <- which(data$unproductive_after < 0) # hay un error pero no entiendo la razon! 
data <- data[-err, ]

# filter more unreasonable values based on new metrics
data <- data %>% 
  filter(idle_after < 60, idle_after > 0.1) %>%
  filter(duration_next < 120, duration_next > 1) %>%
  filter(fare_next < 200) %>% 
  filter(productivity < 125)

# time discretization
split_time <- function(time) {
  w <- lubridate::wday(time)
  h <- lubridate::hour(time)
  24 * (w - 1) + h + 1
}
split_time_label <- function(time) {
  wdaynames = c("Sun", "Mon", "Tue",
            "Wed", "Thu", "Fri", "Sat")
  w = wdaynames[lubridate::wday(time)]
  h = sprintf("%02d:00", lubridate::hour(time))
  paste(w, h)
}
data$timebin <- split_time(data$completed_on)
data$timelabel <- split_time_label(data$completed_on)

# define a column to indicate the spatio-temporal bin
data$node <- paste(data$end_taz, data$timebin, sep="-")

# save processed data
write_csv(data, "./processed_data/rideaustin_productivity.csv")

# quantiles
levs <-  5
x <- data$productivity
splits <- quantile(x, seq(2^(-6), 1 - 2^(-6), length.out=2^5 - 1))
splits <- as.numeric(c(0, splits, 125))
write_csv(data.frame(splits), "./processed_data/splitlevels.csv", col_names=FALSE)
# hist(x)
# abline(v=splits, col="blue", lty=2)

# Part II. Graph creation

# function for polygon adjacency based on C++ kernel for speed
# function to find all polygon intersects from polygon list
print("finding polygon adjacency....")
Rcpp::sourceCpp("polygon_intersect.cpp")
find_polygon_adjacency <- function(polygons, taz_id, eps=0.000001) {
  N <- length(polygons)
  pb <- progress_bar$new(total = as.integer(N * (N - 1) / 2))
  edges_taz <- list()
  for (i in 1:(N - 1)) {
    taz1 = taz_id[i]
    coords1 <- polygons[[i]]@Polygons[[1]]@coords
    labpt1 <- polygons[[i]]@labpt
    for (j in (i + 1):N) {
      taz2 = taz_id[j]
      coords2 <- polygons[[j]]@Polygons[[1]]@coords
      labpt2 <- polygons[[j]]@labpt
      # do not test pts that are very far to save computation
      labpt_dist <- sqrt(sum((labpt1 - labpt2)^2))
      if (labpt_dist < 0.25) {
        intersected <- polygon_intersect(coords1, coords2, eps)
        if (intersected) {
          edges_taz[[length(edges_taz) + 1]] <- c(taz1, taz2)
        }  
      }
      pb$tick()
    }
  }
  edges_taz
}
# run function and save data frame
taz_id = taz@data$ID
polygons = taz@polygons
edges <- find_polygon_adjacency(polygons, taz_id, eps=0.1)
edges_df = edges %>% 
  map(~tibble(taz1=.x[1], taz2=.x[2])) %>% 
  bind_rows() %>% 
  arrange(taz1, taz2)  
write_csv(edges_df, "./processed_data/taz_adjacency.csv", col_names=FALSE)


