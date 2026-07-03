 CPU parallelism, which is a much simpler lift:

  - Each parameter combination in a window is completely independent — perfect for concurrent.futures.ProcessPoolExecutor
  - Each window is also independent (in sliding mode) — could parallelise across windows too
  - No new dependencies, works on any hardware, could easily deliver 4-16× speedup on a modern laptop

Worth adding as a fast-follow once the core implementation is working —
  it'd be a clean addition since the architecture already isolates each window's backtest into its own runner call.
