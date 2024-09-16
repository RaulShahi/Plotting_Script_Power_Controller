# def add_grid_lines_for_scatter_plot(ax, df):
#     df_sorted = df.sort_values(by='time', ascending=True)
#     prev_mode = None
#     line_positions = []

#     for idx, row in df_sorted.iterrows():
#         current_mode = row['mode']
#         current_time = row['time']

#         if current_mode != prev_mode:
#             if prev_mode is not None:
#                 line_positions.append(current_time)  # Add line at mode change
#             prev_mode = current_mode

#     first_time = df_sorted['time'].iloc[0]
#     last_time = df_sorted['time'].iloc[-1]

#     line_positions.insert(0, first_time)  # Add line at start
#     line_positions.append(last_time)  # Add line at end

#     # Draw lines on scatter plot
#     for pos in sorted(set(line_positions)):
#         ax.axvline(x=pos, color='black', linestyle='--', linewidth=1)

#     return sorted(set(line_positions))


# def add_grid_lines_for_boxplot(ax, df, bin_edges):
#     df_sorted = df.sort_values(by='time', ascending=True)
#     prev_mode = None
#     last_time = df_sorted['time'].iloc[-1]
#     line_positions = []

#     for idx, row in df_sorted.iterrows():
#         current_mode = row['mode']
#         current_time = row['time']

#         if current_mode != prev_mode:
#             if prev_mode is not None:
#                 start_bin = next((i for i, edge in enumerate(bin_edges) if current_time < edge), None)
#                 if start_bin is not None and start_bin > 0:
#                     # ax.axvline(x=start_bin - 1, color='black', linestyle='--', linewidth=1)
#                     line_positions.append(start_bin - 1)
#             prev_mode = current_mode

#     end_bin = next((i for i, edge in enumerate(bin_edges) if last_time < edge), None)
#     if end_bin is not None and end_bin > 0:
#         # ax.axvline(x=end_bin - 1, color='black', linestyle='--', linewidth=1)
#         line_positions.append(end_bin - 1)

#     return line_positions

#CSV setup
	# 	timestamp = int(time.time())
	# 	power_mode = rc_opts.get('power_mode', 'default')
	# 	current_directory = os.path.dirname(__file__)
	# 	data_dir = rc_opts.get('data_dir', current_directory)
	# 	self.csv_file = os.path.join(data_dir, f"{timestamp}_{power_mode}_mode_expected_throughput.csv")
	# 	try:
	# 		with open(self.csv_file, mode='w', newline='') as file:
	# 			writer = csv.writer(file, delimiter=' ')
	# 			writer.writerow(["time", "throughput"])
	# 			# self._log.info(f"CSV file created at: {self.csv_file}")
	# 	except Exception as e:
	# 		self._log.info(f"Failed to create CSV file: {e}")

	# def write_to_csv(self, throughput):
	# 	"""Write the current throughput and elapsed time to the CSV file."""
	# 	try:
	# 		elapsed_time_ms = time.time() * 1000 - self._start_time_ms
	# 		# Append timestamp and throughput to the CSV file
	# 		with open(self.csv_file, mode='a', newline='') as file:
	# 			writer = csv.writer(file, delimiter=' ')
	# 			writer.writerow([round(elapsed_time_ms, 3), throughput])
	# 			# self._log.info(f"Written to CSV: {elapsed_time_ms} ms, {throughput}")
	# 	except Exception as e:
	# 			self._log.info(f"Failed to write to CSV file: {e}")