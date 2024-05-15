import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# Set random seed for reproducibility
np.random.seed(0)

# Generate dates
dates = np.arange('2023-01-01', '2023-12-30', dtype='datetime64[D]')

# Generate variable amplitude and phase shift
amplitudes = np.random.normal(1.0, 0.7, size=dates.size)
phase_shifts = np.cumsum(np.random.normal(0, 0.2, size=dates.size))

# Generate base sine wave with varying amplitude and phase
base_wave = amplitudes * np.sin(2 * np.pi * dates.astype(float) / 30 + phase_shifts)

# Gaussian noise
noise = np.random.normal(0, 0.2, dates.size)
# Occasional spikes
spike_noise = np.zeros(dates.size)
spike_indices = np.random.choice(dates.size, size=10, replace=False)
spike_noise[spike_indices] = np.random.normal(0, 1, size=10)

# Combine base wave and noise
prices = base_wave + noise + spike_noise


# Finding extremely significant maxima
extremely_significant_maxima_indices, _ = find_peaks(prices, prominence=1.5)  # Significantly increase prominence
# Finding extremely significant minima by inverting the data
extremely_significant_minima_indices, _ = find_peaks(-prices, prominence=1.5)  # Significantly increase prominence

# Get the values at these extremely significant points
extremely_significant_maxima_dates = dates[extremely_significant_maxima_indices]
extremely_significant_maxima_prices = prices[extremely_significant_maxima_indices]
extremely_significant_minima_dates = dates[extremely_significant_minima_indices]
extremely_significant_minima_prices = prices[extremely_significant_minima_indices]

first_point_date = dates[0]
first_point_price = prices[0]
last_point_date = dates[-1]
last_point_price = prices[-1]

# Combine these points and sort them
all_extremely_significant_points = np.concatenate(([first_point_date], extremely_significant_maxima_dates, 
                                extremely_significant_minima_dates, [last_point_date]))
all_extremely_significant_prices = np.concatenate(([first_point_price], extremely_significant_maxima_prices, 
                                extremely_significant_minima_prices, [last_point_price]))
extremely_significant_sorted_indices = np.argsort(all_extremely_significant_points)
all_extremely_significant_points_sorted = all_extremely_significant_points[extremely_significant_sorted_indices]
all_extremely_significant_prices_sorted = all_extremely_significant_prices[extremely_significant_sorted_indices]


# Plotting the new graph
plt.figure(figsize=(14, 7))
plt.plot(dates, prices, label='Original Price', alpha=0.3, linewidth=1)
plt.scatter(all_extremely_significant_points_sorted, all_extremely_significant_prices_sorted, color='green', s=10, label='Extremely Significant Maxima and Minima')
plt.plot(all_extremely_significant_points_sorted, all_extremely_significant_prices_sorted, 'r', linewidth=1.5)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.title('Further Reduced Zigzag Pattern of Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

