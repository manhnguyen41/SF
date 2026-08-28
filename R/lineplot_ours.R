# Load necessary libraries
library(ggplot2)    # For plotting
library(reshape2)   # For reshaping data
library(wesanderson) # For color palettes
library(dplyr)      # For data manipulation

# --- 1. Read data from separate CSV files ---
df_our_model <- read.csv("data/our_model.csv", header = TRUE, sep = ",")
df_groundtruth <- read.csv("data/groundtruth.csv", header = TRUE, sep = ",")

# Đổi tên cột giá trị: Thay "Prediction" bằng "Ours"
colnames(df_our_model)[2] <- "Ours"
colnames(df_groundtruth)[2] <- "Groundtruth"

# Gộp dữ liệu dựa trên cột 'index'
combined_data <- inner_join(df_groundtruth, df_our_model, by = "index")

# Create a data frame with Groundtruth and Ours
data_plot <- data.frame(
  Index = combined_data$index, 
  Groundtruth = combined_data$Groundtruth,
  Ours = combined_data$Ours
)

# --- 2. Define Plot Parameters ---
y_min <- 0
y_max <- 300

# Convert data to long format
data_long <- melt(data_plot, id.vars = "Index", 
                  variable.name = "Type", value.name = "Value")

begin_index <- 0 
end_index <- max(data_plot$Index) 

# Get specific colors from the "Darjeeling1" palette
darjeeling_colors <- wes_palette("Darjeeling1")

# CẬP NHẬT TÊN Ở ĐÂY: Thay "Prediction" thành "Ours"
specific_colors <- c("Groundtruth" = darjeeling_colors[5], "Ours" = darjeeling_colors[1])
specific_shapes <- c("Groundtruth" = 16, "Ours" = 16) 

text_size <- 26

# --- 3. Custom Theme ---
custom_theme <- theme_minimal() +
  theme(
    panel.border = element_rect(colour = "black", fill = NA, size = 1),
    panel.grid.major = element_line(size = 0.5, linetype = "dashed", color = "gray80"),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = text_size, face = "bold"),
    axis.text = element_text(size = text_size - 2),
    legend.position = "top",
    legend.text = element_text(size = text_size - 2),
    legend.title = element_blank(),
    legend.background = element_rect(colour = "black", fill = "white", size = 0.5),
    plot.margin = unit(c(1, 1, 1, 1), "cm")
  )

# --- 4. Create the Plot ---
plot <- ggplot(data_long, aes(x = Index, y = Value, color = Type, shape = Type)) + 
  geom_line(size = 1) + 
  geom_point(size = 2) + 
  labs(
    x = "Index", 
    y = "Rainfall", 
    color = "Data Type", 
    shape = "Data Type"
  ) +
  scale_x_continuous(limits = c(begin_index, end_index), 
                     breaks = seq(begin_index, end_index, by = 30)) + 
  scale_y_continuous(limits = c(y_min, y_max)) + 
  scale_color_manual(values = specific_colors) + 
  scale_shape_manual(values = specific_shapes) + 
  custom_theme

# Display the plot
print(plot)

# --- 5. Save the plot ---
output_dir <- "Fig/4lineplot/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

ggsave(file.path(output_dir, "ours_comparison.eps"), plot = plot, width = 8, height = 6, units = "in", device = "eps")
ggsave(file.path(output_dir, "ours_comparison.png"), plot = plot, width = 8, height = 6, units = "in", device = "png", dpi = 300)