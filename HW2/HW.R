toy_data <- function(n, seed = NULL) {
  set.seed(seed)
  x <- matrix(rnorm(8 * n), ncol = 8)
  z <- 0.4 * x[, 1] - 0.5 * x[, 2] + 1.75 * x[, 3] - 0.2 * x[, 4] + x[, 5]
  y <- runif(n) > 1 / (1 + exp(-z))
  return(data.frame(x = x, y = y))
}

log_loss <- function(y, p) {
  -(y * log(p) + (1 - y) * log(1 - p))
}

sample_size <- 100000
df_dgp <- toy_data(sample_size, 0)


z_val <- qnorm(0.05 / 2, lower.tail = FALSE)

# Task 1
train_data <- toy_data(50, 0)

model <- glm(train_data$y ~ ., data = train_data, family = binomial)

predictions <- predict(model, newdata = df_dgp, type = "response")

true_risk <- sum(log_loss(df_dgp$y, predictions)) / sample_size

risk_in_interval <- 0

run_times <- 1000

test_risks <- c()

for (i in seq(1:run_times + 1)) {
  test_data <- toy_data(50, i)
  test_pred <- predict(model, newdata = test_data, type = "response")
  test_losses <- log_loss(test_data$y, test_pred)
  test_risk <- sum(test_losses) / nrow(test_data)

  test_risks <- c(test_risks, test_risk)

  sample_stderr <- sd(test_losses) / sqrt(length(test_losses))

  t_score <- qt(0.025, df = length(test_losses) - 1, lower.tail = FALSE)

  # Margin of error
  moe <- t_score * sample_stderr

  ci_lower <- test_risk - moe
  ci_upper <- test_risk + moe

  if (true_risk > ci_lower && true_risk < ci_upper) {
    risk_in_interval <- risk_in_interval + 1
  }
}

plot_risks <- test_risks - true_risk
plot_risks_density <- density(plot_risks)

plot(plot_risks_density, main = "Density Plot",
     xlab = "test_risk - true_risk", ylab = "Density")


percentage <- risk_in_interval / run_times
print(percentage)
