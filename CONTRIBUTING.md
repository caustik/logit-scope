# Contributing

Issues and focused pull requests are welcome. Please keep the application local-first, avoid adding a mandatory frontend toolchain, and preserve candidate rank unless an experimental mode is clearly labeled otherwise.

Before submitting a change:

1. Build the `logit-scope` target.
2. Run `ctest --output-on-failure` for your configuration.
3. If sampling changed, exercise at least one real GGUF chat response with the None profile and with one shaping profile enabled.
4. If the UI changed, check it at desktop and narrow viewport widths.

Do not include model files, generated build output, or local conversation data.
