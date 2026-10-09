# magika-burn

[Magika](https://github.com/google/magika) is Google's deep-learning model for
detecting file content types. **magika-burn** runs it in pure Rust: the model's
forward pass is written out in plain Rust, with its weights embedded, so there
is no ONNX Runtime, no machine-learning framework, no C++ and no Python. The
same code runs natively and as WebAssembly in browsers, Node.js and Bun.

This is an unofficial port and is not affiliated with Google. For the official
implementations, see [google/magika](https://github.com/google/magika).

## Installation

Rust (1.92 or later):

```sh
cargo add magika-burn
```

JavaScript:

```sh
npm install @yorganci/magika-burn
```

Command line (installs `magika`):

```sh
cargo install --git https://github.com/atahanyorganci/magika-burn magika-burn-cli
```

## Usage

### Rust

```rust no_run
use std::fs::File;

use magika_burn::Magika;

fn main() -> std::io::Result<()> {
    let magika = Magika::new();
    let prediction = magika.identify_reader(File::open("report.pdf")?)?;
    let info = prediction.info();
    println!("{} ({}), score {:.3}", info.label, info.mime_type, prediction.score);
    Ok(())
}
```

`identify_bytes` works on in-memory content. The prediction mode sets how
confident the model must be (see [Results](#results)), and content types can be
looked up without running the model:

```rust
use magika_burn::{ContentType, Magika, PredictionMode};

let magika = Magika::new().with_prediction_mode(PredictionMode::BestGuess);
let prediction = magika.identify_bytes(b"#!/bin/sh\necho 'Hello, world!'\n");
println!("{}: {}", prediction.output, prediction.info().description);

assert_eq!(magika.identify_bytes(b"").output, ContentType::Empty);

let pdf = ContentType::from_label("pdf").unwrap();
assert_eq!(pdf, ContentType::Pdf);
assert_eq!(pdf.info().mime_type, "application/pdf");
assert_eq!(pdf.info().extensions, ["pdf"]);
```

The `serde` feature implements `Serialize` and `Deserialize` for the public
types, with content types as their labels. A serialized `Prediction` has the same
shape as upstream's JSON output.

The crate also builds for `wasm32-unknown-unknown`. Enable WebAssembly SIMD with
`RUSTFLAGS="-C target-feature=+simd128"` for faster inference.

### Node.js

```js
import { readFileSync } from "node:fs";
import { MagikaModel } from "@yorganci/magika-burn";

const magika = new MagikaModel();
const { output, score } = magika.identifyBytes(readFileSync("report.pdf"));
console.log(output.label, output.mimeType, score);
```

### Browsers and bundlers

The browser build has to be initialized before use. `init()` fetches
`magika_bg.wasm` from next to the JavaScript module:

```js
import init, { MagikaModel } from "@yorganci/magika-burn";

await init();
const magika = new MagikaModel({ predictionMode: "high_confidence" });

input.addEventListener("change", async () => {
  const content = new Uint8Array(await input.files[0].arrayBuffer());
  const { output } = magika.identifyBytes(content);
  console.log(`${output.description} (${output.mimeType})`);
});
```

If your bundler doesn't handle `new URL("…", import.meta.url)`, pass the URL or
the bytes of `@yorganci/magika-burn/magika_bg.wasm` to
`init({ module_or_path })` instead.

Identifying a file takes about 20 ms, so run the model in a Web Worker if you
identify many files. `MagikaModel` is a small object in WebAssembly memory,
which `magika.free()` releases, as does declaring it with `using`.

### Bun

`bun run` uses the Node.js entry point, which reads `magika_bg.wasm` from the
package. A [single-file executable](https://bun.sh/docs/bundler/executables) built
with `bun build --compile` cannot, so embed the WebAssembly module and pass it
to `initSync` from the browser build:

```js
import { readFileSync } from "node:fs";
import wasm from "@yorganci/magika-burn/magika_bg.wasm" with { type: "file" };
import { initSync, MagikaModel } from "@yorganci/magika-burn/web";

initSync({ module: readFileSync(wasm) });
const magika = new MagikaModel();
```

### Command line

`magika` identifies files, directories and standard input (`-`), and prints one
line per path. When the model is not confident, the line shows its guess and
score:

```text
$ magika README.md src/lib.rs Cargo.toml data.bin missing.txt
README.md: Markdown document (text)
src/lib.rs: Rust source (code)
Cargo.toml: Tom's obvious, minimal language (text)
data.bin: Unknown binary data (unknown) [low confidence: Web Assembly (executable), 31%]
missing.txt: error: No such file or directory (os error 2)
```

| Option                           | Description                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| `-r`, `--recursive`              | Identify the files inside directories instead of the directories themselves |
| `--no-dereference`               | Identify symbolic links as such instead of following them                   |
| `-m`, `--prediction-mode <MODE>` | `high-confidence` (default), `medium-confidence` or `best-guess`            |
| `-l`, `--label`                  | Print the label (`rust`) instead of the description                         |
| `-i`, `--mime-type`              | Print the MIME type (`application/x-rust`) instead of the description       |
| `-s`, `--output-score`           | Append the score                                                            |
| `-j`, `--jobs <N>`               | Identify this many files at a time, by default one per processor            |
| `--json`                         | Print a JSON array with the prediction and content type of each path        |
| `--list-content-types`           | List all content types, as a table or with `--json` as JSON                 |

`magika` exits with 1 if a path could not be identified, and with 2 for invalid
usage.

## Results

A prediction contains two content types:

| Rust               | JavaScript        | Meaning                                                                           |
| ------------------ | ----------------- | --------------------------------------------------------------------------------- |
| `dl`               | `dl`              | The model's prediction, or the `undefined` content type if the model was not run. |
| `output`           | `output`          | The final content type, after the overwrite map and the confidence threshold.     |
| `score`            | `score`           | The model's score for `dl`, or 1 if the model was not run.                        |
| `overwrite_reason` | `overwriteReason` | Why `output` differs from `dl`.                                                   |

In Rust `dl` and `output` are `ContentType`s, and `.info()` returns their
`ContentTypeInfo`. In JavaScript they are `ContentTypeInfo` objects:
`{ label, mimeType, group, description, extensions, isText }`.

The prediction mode sets the confidence threshold. Below it, `output` becomes
`txt` for text content types and `unknown` for binary ones:

| Mode                                             | Threshold                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `HighConfidence` / `"high_confidence"` (default) | Per content type, e.g. 0.75 for `markdown` and 0.95 for `latex`, and 0.5 for the rest |
| `MediumConfidence` / `"medium_confidence"`       | 0.5                                                                                   |
| `BestGuess` / `"best_guess"`                     | None                                                                                  |

The overwrite reason is one of:

- `None`: `output` is `dl`.
- `LowConfidence`: the score is below the threshold.
- `OverwriteMap`: the model's label is replaced by a more generic one. The model
  recognizes random bytes and random text, which are reported as `unknown` and
  `txt`.

The model is not run for empty content (`empty`), or for content with fewer than
8 bytes besides leading and trailing whitespace. That content is `txt` if it is
valid UTF-8 and `unknown` otherwise.

The Rust crate also exposes the individual steps: `Features::extract`,
`Magika::scores` and `Prediction::from_scores`.

## How it works

1. **Features:** Magika reads up to 4 KiB from each end of the content. It strips
   whitespace and keeps the first and last 1024 bytes, padding shorter content.
2. **Model:** Magika's `standard_v3_3` model is an embedding of the bytes, a
   layer norm, a convolution, a second layer norm and a dense layer. Its forward
   pass is written out for f32 in [`src/model.rs`](src/model.rs), with the
   weights embedded in the binary. The convolution takes about 95% of the time.
   [`scripts/prepare_model.py`](scripts/prepare_model.py) extracts the weights
   from the ONNX model and checks a NumPy version of the forward pass against
   ONNX Runtime; the tests check the Rust one.
3. **Content types:** the scores are resolved with the thresholds and the
   overwrite map from the model's configuration. MIME types and other information
   come from upstream's content types knowledge base.

The upstream files are vendored in [`assets/`](assets); see
[`assets/README.md`](assets/README.md). Compared to upstream Magika, this port
does not include the optional magic-byte rules. It identifies bytes and readers,
not paths, so there is no handling of directories or symbolic links.

## Performance

Identifying one file at a time on an Apple M3 Max:

| Implementation                           | Time per file | Memory  |
| ---------------------------------------- | ------------- | ------- |
| Upstream Magika (ONNX Runtime)           | ~2.5 ms       |         |
| magika-burn, native (one thread)         | ~14 ms        |         |
| magika-burn, WebAssembly in Node.js 24   | ~18 ms        | ~13 MiB |
| magika-burn, WebAssembly in Bun 1.4      | ~18 ms        | ~26 MiB |

Memory is the resident memory the WebAssembly module adds, including compiling
it, as measured by `scripts/measure-wasm.mjs`. The WebAssembly module is 3.2 MB
(3.0 MB gzipped), almost all of it weights: its code is 36 KiB.

The times above are for one file on one thread. `magika` identifies several at a
time, one per processor by default and `--jobs` otherwise, and `Magika` is `Sync`,
so one model can be shared by as many threads as you like. On x86-64 the model
also picks a forward pass compiled for AVX2 and FMA when the processor has them.

## Development

`nix develop` provides the Rust toolchain with the WebAssembly target,
`wasm-bindgen-cli`, Node.js, pnpm and Bun.

```sh
cargo test --workspace --all-features  # Rust and CLI tests, including upstream reference data
pnpm run build                         # build pkg/web
pnpm test                              # test the WebAssembly build with Node.js
tests/bun/compile.sh                   # test a compiled Bun executable
bun scripts/measure-wasm.mjs           # measure memory and time (or node --expose-gc)
uv run scripts/prepare_model.py        # re-vendor upstream assets and regenerate fixtures
```

## License

Copyright 2026 Atahan Yorgancı. Licensed under the [Apache License 2.0](LICENSE).

The Magika model, its configuration, the content types knowledge base and the
upstream test data are © Google LLC and also licensed under the Apache License
2.0 ([`assets/LICENSE`](assets/LICENSE)). `weights.f32` is extracted from the
model as described in [`assets/README.md`](assets/README.md).
