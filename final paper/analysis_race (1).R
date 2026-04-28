# ---- Setup ----

library(dplyr)
library(ggplot2)
library(nnet)
library(randomForest)
library(gbm)
library(pROC)
library(tidyr)
library(patchwork)
library(kableExtra)
library(scales)

set.seed(42)

race_colors  <- c("White" = "#1f77b4", "Black" = "#d62728", "Hispanic/Latino" = "#2ca02c")
model_colors <- c("LR" = "#1f77b4", "RF" = "#2ca02c", "GB" = "#d62728")


# ---- Load and clean data ----

raw <- readLines("police_killings.csv", encoding = "latin1", warn = FALSE)
df  <- read.csv(text = raw, stringsAsFactors = FALSE)

num_cols <- c("age","share_white","share_black","share_hispanic",
              "p_income","pov","h_income","urate","college","pop")
df[num_cols] <- lapply(df[num_cols], function(x) suppressWarnings(as.numeric(x)))

df <- df %>%
  filter(raceethnicity %in% c("White","Black","Hispanic/Latino"))

df$armed_group <- ifelse(df$armed == "Firearm", "Firearm",
                  ifelse(df$armed == "No",      "Unarmed",
                                                "Other_weapon"))

req <- c("age","pop","h_income","pov","urate","college",
         "share_white","share_black","share_hispanic",
         "gender","cause","armed_group","raceethnicity")
df <- df[complete.cases(df[, req]), ]

df$gender        <- as.factor(df$gender)
df$cause         <- as.factor(df$cause)
df$armed_group   <- as.factor(df$armed_group)
df$raceethnicity <- factor(df$raceethnicity,
                           levels = c("White","Black","Hispanic/Latino"))

cat(sprintf("Analysis sample: n = %d\n", nrow(df)))
print(table(df$raceethnicity))


# ---- Define feature sets ----

formula_A <- raceethnicity ~ age + gender + cause + armed_group
formula_B <- raceethnicity ~ h_income + pov + urate + college
formula_C <- raceethnicity ~ age + gender + cause + armed_group +
                              h_income + pov + urate + college
formula_D <- raceethnicity ~ age + gender + cause + armed_group +
                              h_income + pov + urate + college +
                              share_white + share_black + share_hispanic

feature_sets <- list("A. Incident only"        = formula_A,
                     "B. Tract SES only"       = formula_B,
                     "C. Incident + SES"       = formula_C,
                     "D. C + Tract racial mix" = formula_D)


# ---- Cross-validation setup ----

make_folds <- function(y, k = 5, seed = 42) {
  set.seed(seed)
  folds <- integer(length(y))
  for (lvl in levels(y)) {
    idx <- which(y == lvl)
    folds[idx] <- sample(rep(1:k, length.out = length(idx)))
  }
  folds
}
folds <- make_folds(df$raceethnicity, k = 5)

macro_ovr_auc <- function(y_true, prob_matrix) {
  classes <- colnames(prob_matrix)
  aucs <- sapply(classes, function(c) {
    y_bin <- as.integer(y_true == c)
    as.numeric(pROC::auc(y_bin, prob_matrix[, c], quiet = TRUE))
  })
  mean(aucs)
}


# ---- Fit models per fold ----

fit_fold <- function(train, test, f) {
  lr_fit <- nnet::multinom(f, data = train, trace = FALSE, maxit = 300)
  lr_prob <- predict(lr_fit, test, type = "probs")
  lr_pred <- predict(lr_fit, test, type = "class")

  rf_fit <- randomForest(f, data = train, ntree = 500,
                         nodesize = 5, maxnodes = 64)
  rf_prob <- predict(rf_fit, test, type = "prob")
  rf_pred <- predict(rf_fit, test, type = "response")

  classes <- levels(train$raceethnicity)
  gb_prob <- matrix(0, nrow = nrow(test), ncol = length(classes),
                    dimnames = list(NULL, classes))
  for (c in classes) {
    train_c <- train
    train_c$y_c <- as.integer(train$raceethnicity == c)
    f_gb <- update(f, y_c ~ . - raceethnicity)
    gb_fit <- gbm(f_gb, data = train_c, distribution = "bernoulli",
                  n.trees = 200, interaction.depth = 3,
                  shrinkage = 0.05, verbose = FALSE)
    gb_prob[, c] <- predict(gb_fit, test, n.trees = 200, type = "response")
  }
  gb_prob <- gb_prob / rowSums(gb_prob)
  gb_pred <- factor(classes[apply(gb_prob, 1, which.max)], levels = classes)

  list(LR = list(prob = lr_prob, pred = lr_pred),
       RF = list(prob = rf_prob, pred = rf_pred),
       GB = list(prob = gb_prob, pred = gb_pred))
}


# ---- Run CV across all feature sets ----

run_cv <- function(f) {
  classes <- levels(df$raceethnicity); n <- nrow(df)
  prob_all <- list(LR = matrix(NA, n, 3, dimnames = list(NULL, classes)),
                   RF = matrix(NA, n, 3, dimnames = list(NULL, classes)),
                   GB = matrix(NA, n, 3, dimnames = list(NULL, classes)))
  pred_all <- list(LR = factor(rep(NA, n), levels = classes),
                   RF = factor(rep(NA, n), levels = classes),
                   GB = factor(rep(NA, n), levels = classes))

  for (k in 1:5) {
    train <- df[folds != k, ]; test <- df[folds == k, ]
    fits <- fit_fold(train, test, f)
    for (m in names(fits)) {
      prob_all[[m]][folds == k, ] <- fits[[m]]$prob
      pred_all[[m]][folds == k]   <- fits[[m]]$pred
    }
  }
  list(prob = prob_all, pred = pred_all)
}

cv_results <- list()
for (nm in names(feature_sets)) {
  cat(sprintf("Running CV for feature set: %s\n", nm))
  cv_results[[nm]] <- run_cv(feature_sets[[nm]])
}


# ---- Summarize performance ----

y_true <- df$raceethnicity

perf <- data.frame()
for (nm in names(cv_results)) {
  for (m in c("LR","RF","GB")) {
    acc <- mean(cv_results[[nm]]$pred[[m]] == y_true)
    auc <- macro_ovr_auc(y_true, cv_results[[nm]]$prob[[m]])
    perf <- rbind(perf, data.frame(feature_set = nm, model = m,
                                    accuracy = acc, auc = auc))
  }
}
majority <- max(table(y_true)) / length(y_true)
cat(sprintf("\nMajority-class baseline accuracy: %.3f\n", majority))
print(perf)


# ---- Paired bootstrap C vs D ----

correct_C <- as.integer(cv_results[["C. Incident + SES"]]$pred$LR == y_true)
correct_D <- as.integer(cv_results[["D. C + Tract racial mix"]]$pred$LR == y_true)

boot_diffs <- replicate(2000, {
  i <- sample.int(length(y_true), replace = TRUE)
  mean(correct_D[i]) - mean(correct_C[i])
})
p_boot <- 2 * min(mean(boot_diffs <= 0), mean(boot_diffs >= 0))
cat(sprintf("\nLR paired bootstrap D - C: mean gain = %+.3f, 95%% CI [%+.3f, %+.3f], p = %.4f\n",
            mean(boot_diffs), quantile(boot_diffs, 0.025),
            quantile(boot_diffs, 0.975), p_boot))


# ---- Per-class AUC ----

per_class <- data.frame()
for (nm in names(cv_results)) {
  prob <- cv_results[[nm]]$prob$LR
  for (c in levels(y_true)) {
    y_bin <- as.integer(y_true == c)
    auc_c <- as.numeric(pROC::auc(y_bin, prob[, c], quiet = TRUE))
    per_class <- rbind(per_class,
                       data.frame(feature_set = nm, race = c, auc = auc_c))
  }
}
print(per_class)


# ---- One-vs-rest LR coefficients (Model C) ----

df_std <- df
num_to_scale <- c("age","h_income","pov","urate","college")
df_std[, num_to_scale] <- scale(df[, num_to_scale])

races_to_fit <- c("White","Black","Hispanic/Latino")
non_race_features <- c("age","h_income","pov","urate","college")

coefs_C <- matrix(0, nrow = length(non_race_features),
                  ncol = length(races_to_fit),
                  dimnames = list(non_race_features, races_to_fit))

for (r in races_to_fit) {
  df_std$y_ovr <- as.integer(df_std$raceethnicity == r)
  fit_ovr <- glm(y_ovr ~ age + h_income + pov + urate + college,
                 data = df_std, family = binomial)
  coefs_C[, r] <- coef(fit_ovr)[non_race_features]
}

cat("\nStandardized one-vs-rest LR coefficients (Model C):\n")
print(round(coefs_C, 3))


# ---- RF permutation importance (Model D) ----

rf_D_full <- randomForest(formula_D, data = df, ntree = 500,
                          nodesize = 5, maxnodes = 64,
                          importance = TRUE)
imp_df <- data.frame(feature    = rownames(importance(rf_D_full)),
                     importance = importance(rf_D_full)[, "MeanDecreaseAccuracy"])
imp_df <- imp_df[order(-imp_df$importance), ]
print(imp_df)


# ---- Plot theme and helpers ----

theme_report <- function() {
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          plot.title       = element_text(face = "bold", size = 12))
}

pretty_names <- c(share_black="Tract % Black", share_white="Tract % White",
                  share_hispanic="Tract % Hispanic", age="Victim age",
                  college="Tract % college", urate="Unemployment rate",
                  pov="Poverty rate", h_income="Household income",
                  pop="Tract population", cause="Cause of death",
                  armed_group="Armed status", gender="Gender")


# ---- Figure 1: accuracy and AUC across feature ladder ----

perf_long <- perf %>%
  mutate(set_short = substr(feature_set, 1, 1))

fig1a <- ggplot(perf_long, aes(x = set_short, y = accuracy, fill = model)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = majority, linetype = "dashed", color = "grey30") +
  annotate("text", x = 0.8, y = majority + 0.01,
           label = sprintf("Majority baseline (%.2f)", majority),
           hjust = 0, size = 3, color = "grey30") +
  scale_fill_manual(values = model_colors) +
  labs(title = "Accuracy across feature ladder",
       x = "Feature set", y = "Accuracy (5-fold CV)", fill = "Model") +
  coord_cartesian(ylim = c(0.4, 0.8)) +
  theme_report()

fig1b <- ggplot(perf_long, aes(x = set_short, y = auc, fill = model)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey30") +
  annotate("text", x = 0.8, y = 0.51, label = "Chance (0.5)",
           hjust = 0, size = 3, color = "grey30") +
  scale_fill_manual(values = model_colors) +
  labs(title = "Macro one-vs-rest AUC across feature ladder",
       x = "Feature set", y = "AUC", fill = "Model") +
  coord_cartesian(ylim = c(0.45, 0.9)) +
  theme_report()

fig1 <- fig1a + fig1b + plot_layout(guides = "collect")
print(fig1)


# ---- Figure 2: per-class AUC across feature ladder ----

per_class_plot <- per_class %>%
  mutate(set_short = substr(feature_set, 1, 1),
         race = factor(race, levels = c("White","Black","Hispanic/Latino")))

fig2 <- ggplot(per_class_plot, aes(x = set_short, y = auc, fill = race)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey30") +
  scale_fill_manual(values = race_colors) +
  labs(title = "Per-class prediction accuracy by feature set",
       x = "Feature set", y = "Per-class AUC (one-vs-rest)", fill = "Race") +
  coord_cartesian(ylim = c(0.5, 0.9)) +
  theme_report()
print(fig2)


# ---- Figure 3: confusion matrices for Models C and D ----

make_cm_df <- function(pred, truth, set_name) {
  tab <- table(truth, pred)
  as.data.frame(tab, responseName = "count") %>%
    group_by(truth) %>%
    mutate(pct = count / sum(count),
           set = set_name) %>%
    ungroup()
}

cm_C <- make_cm_df(cv_results[["C. Incident + SES"]]$pred$LR, y_true,
                   "Model C (non-race features only)")
cm_D <- make_cm_df(cv_results[["D. C + Tract racial mix"]]$pred$LR, y_true,
                   "Model D (adds tract racial mix)")
cm_both <- bind_rows(cm_C, cm_D)

fig3 <- ggplot(cm_both, aes(x = pred, y = truth, fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%d\n(%.0f%%)", count, pct * 100)),
            size = 3.3) +
  facet_wrap(~ set) +
  scale_fill_gradient(low = "#f0f5fb", high = "#1f77b4",
                      labels = scales::percent_format(accuracy = 1),
                      guide = "none") +
  scale_x_discrete(position = "top") +
  labs(title = "Confusion matrices for Models C and D",
       x = "Predicted race", y = "Actual race") +
  theme_report() +
  theme(axis.text.x = element_text(angle = 0))
print(fig3)


# ---- Figure 4: feature importance and standardized coefficients ----

imp_df$pretty  <- ifelse(imp_df$feature %in% names(pretty_names),
                         pretty_names[imp_df$feature], imp_df$feature)
imp_df$is_race <- imp_df$feature %in% c("share_white","share_black","share_hispanic")
imp_top <- head(imp_df, 10)

fig4a <- ggplot(imp_top, aes(x = reorder(pretty, importance), y = importance,
                             fill = is_race)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#d62728", "FALSE" = "#888888"),
                    labels = c("TRUE" = "Tract racial composition",
                               "FALSE" = "Non-race feature"),
                    name = NULL) +
  labs(title = "RF permutation importance",
       subtitle = "Model D (race-blind features plus tract racial composition)",
       x = NULL, y = "Mean decrease in accuracy") +
  theme_report() +
  theme(legend.position = "bottom",
        legend.box      = "vertical",
        legend.key.size = unit(0.4, "cm"),
        plot.title    = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 9, color = "grey30")) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

non_race <- c("age","h_income","pov","urate","college")
coef_long <- as.data.frame(coefs_C[non_race, ]) %>%
  tibble::rownames_to_column("feature") %>%
  pivot_longer(-feature, names_to = "race", values_to = "coef") %>%
  mutate(pretty = pretty_names[feature],
         race   = factor(race, levels = c("White","Black","Hispanic/Latino")))

fig4b <- ggplot(coef_long, aes(x = coef, y = reorder(pretty, abs(coef)),
                               fill = race)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.4) +
  scale_fill_manual(values = race_colors) +
  labs(title = "Standardized LR coefficients",
       subtitle = "Model C (non-race features only)",
       x = "Coefficient (positive: more likely)",
       y = NULL, fill = "Predicted race") +
  theme_report() +
  theme(legend.position = "bottom",
        legend.box      = "vertical",
        legend.key.size = unit(0.4, "cm"),
        plot.title    = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 9, color = "grey30")) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

fig4 <- fig4a + fig4b + plot_layout(widths = c(1, 1))
print(fig4)


# ---- Table 1: model comparison ----

table1 <- perf %>%
  pivot_wider(names_from = model, values_from = c(accuracy, auc)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

baseline_row <- tibble::tibble(
  feature_set = "[Majority-class baseline]",
  accuracy_LR = round(majority, 3),
  auc_LR = NA, accuracy_RF = NA, auc_RF = NA,
  accuracy_GB = NA, auc_GB = NA)
table1 <- bind_rows(baseline_row, table1)

table1 %>%
  kbl(caption = "Table 1. Cross-validated performance across feature sets and models.",
      col.names = c("Feature set", "LR accuracy", "LR AUC",
                    "RF accuracy", "RF AUC", "GB accuracy", "GB AUC"),
      align = "lrrrrrr") %>%
  kable_styling(bootstrap_options = c("striped","hover"), full_width = FALSE)
