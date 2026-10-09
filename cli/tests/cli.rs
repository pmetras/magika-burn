//! Runs the `magika` binary on files in a temporary directory and snapshots its
//! exit code, standard output and standard error.

use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

use tempfile::TempDir;

const RUST: &str = r#"use std::collections::HashMap;

/// Counts how often each word appears in the input.
fn word_counts(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let counts = word_counts("the quick brown fox jumps over the lazy dog");
    for (word, count) in &counts {
        println!("{word}: {count}");
    }
}
"#;

const MARKDOWN: &str = r#"# Release notes

## Features

- Identify the content type of files with **Magika**.
- Print labels, MIME types or descriptions.

## Usage

```sh
magika README.md
```

See the [documentation](https://github.com/google/magika) for details.
"#;

/// A temporary directory with one file per interesting case.
fn fixtures() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, content: &[u8]| fs::write(dir.path().join(name), content).unwrap();
    write("main.rs", RUST.as_bytes());
    write("notes.md", MARKDOWN.as_bytes());
    // Too short for the model: decided by whether they are UTF-8.
    write("empty.txt", b"");
    write("hello.txt", b"hello\n");
    write("binary.bin", &[0xff, 0xfe, 0xfd, 0xfc]);
    // The model guesses wasm with a low score (upstream reference example).
    write("low-confidence.bin", &[0, 1, 2, 3, 4, 5, 6, 7]);
    fs::create_dir(dir.path().join("directory")).unwrap();
    dir
}

/// Adds `tree/`, with files at two levels, to the fixtures.
fn tree(dir: &Path) {
    fs::create_dir_all(dir.join("tree/sub")).unwrap();
    fs::write(dir.join("tree/main.rs"), RUST).unwrap();
    fs::write(dir.join("tree/hello.txt"), "hello\n").unwrap();
    fs::write(dir.join("tree/sub/empty.txt"), "").unwrap();
    fs::write(dir.join("tree/sub/binary.bin"), [0xff, 0xfe, 0xfd, 0xfc]).unwrap();
}

/// Adds symbolic links to a file, to a missing file, to an ancestor directory
/// (a cycle) and to `tree/`.
#[cfg(unix)]
fn links(dir: &Path) {
    use std::os::unix::fs::symlink;

    tree(dir);
    symlink("hello.txt", dir.join("tree/link.txt")).unwrap();
    symlink("missing.txt", dir.join("tree/broken.txt")).unwrap();
    symlink("..", dir.join("tree/sub/cycle")).unwrap();
    symlink("tree", dir.join("tree-link")).unwrap();
}

/// Runs `magika` in `dir` and formats its exit code and output for a snapshot.
fn magika(dir: &Path, args: &[&str], stdin: &[u8]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_magika"))
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    let output = child.wait_with_output().unwrap();
    format!(
        "exit code: {}\n--- stdout\n{}--- stderr\n{}",
        output.status.code().unwrap(),
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

const ALL: &[&str] = &[
    "main.rs",
    "notes.md",
    "empty.txt",
    "hello.txt",
    "binary.bin",
    "low-confidence.bin",
    "directory",
];

#[test]
fn describes_content_types() {
    let dir = fixtures();
    insta::assert_snapshot!(magika(dir.path(), ALL, b""));
}

#[test]
fn labels() {
    let dir = fixtures();
    let args = [&["--label"], ALL].concat();
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[test]
fn mime_types() {
    let dir = fixtures();
    let args = [&["-i"], ALL].concat();
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[test]
fn scores() {
    let dir = fixtures();
    let args = ["-s", "main.rs", "hello.txt", "low-confidence.bin"];
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[test]
fn prediction_modes() {
    let dir = fixtures();
    let mut output = String::new();
    for mode in ["high-confidence", "medium-confidence", "best-guess"] {
        let args = ["-m", mode, "-l", "low-confidence.bin"];
        output += &magika(dir.path(), &args, b"");
    }
    insta::assert_snapshot!(output);
}

#[test]
fn reads_standard_input() {
    let dir = fixtures();
    let args = ["main.rs", "-", "hello.txt"];
    insta::assert_snapshot!(magika(dir.path(), &args, MARKDOWN.as_bytes()));
}

#[test]
fn reports_errors_and_continues() {
    let dir = fixtures();
    let args = ["missing.txt", "hello.txt"];
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[cfg(unix)]
#[test]
fn rejects_special_files() {
    let dir = fixtures();
    insta::assert_snapshot!(magika(dir.path(), &["/dev/null"], b""));
}

#[test]
fn recurses_into_directories() {
    let dir = fixtures();
    tree(dir.path());
    let args = ["-r", "tree", "hello.txt"];
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

/// Files are identified in parallel but printed in the order they were given, so
/// that a run is reproducible however many of them there are.
#[test]
fn prints_in_the_order_of_the_paths() {
    let dir = fixtures();
    tree(dir.path());
    let mut args = vec!["-r", "tree"];
    args.extend_from_slice(ALL);
    let serial = magika(dir.path(), &[&["-j", "1"], args.as_slice()].concat(), b"");
    for jobs in ["2", "3", "8", "64"] {
        let parallel = magika(dir.path(), &[&["-j", jobs], args.as_slice()].concat(), b"");
        assert_eq!(parallel, serial, "-j {jobs} reordered the output");
    }
}

#[cfg(unix)]
#[test]
fn follows_symbolic_links() {
    let dir = fixtures();
    links(dir.path());
    let mut output = magika(
        dir.path(),
        &["tree-link", "tree/link.txt", "tree/broken.txt"],
        b"",
    );
    output += &magika(dir.path(), &["-r", "tree"], b"");
    insta::assert_snapshot!(output);
}

#[cfg(unix)]
#[test]
fn identifies_symbolic_links_without_dereferencing() {
    let dir = fixtures();
    links(dir.path());
    let args = [
        "--no-dereference",
        "tree-link",
        "tree/link.txt",
        "tree/broken.txt",
    ];
    let mut output = magika(dir.path(), &args, b"");
    output += &magika(dir.path(), &["-r", "--no-dereference", "tree"], b"");
    insta::assert_snapshot!(output);
}

#[test]
fn prints_json() {
    let dir = fixtures();
    // Exact output for content decided without the model (score 1).
    let args = [
        "--json",
        "hello.txt",
        "empty.txt",
        "directory",
        "missing.txt",
    ];
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[test]
fn prints_json_predictions() {
    let dir = fixtures();
    let output = Command::new(env!("CARGO_BIN_EXE_magika"))
        .args(["--json", "main.rs", "low-confidence.bin"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Round scores down to two digits, which are the same on every platform.
    let mut json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for entry in json.as_array_mut().unwrap() {
        let score = &mut entry["prediction"]["score"];
        *score = ((score.as_f64().unwrap() * 100.0).floor() / 100.0).into();
    }
    insta::assert_snapshot!(serde_json::to_string_pretty(&json).unwrap());
}

#[test]
fn prints_an_empty_json_array_without_entries() {
    let dir = fixtures();
    let args = ["--json", "--recursive", "directory"];
    insta::assert_snapshot!(magika(dir.path(), &args, b""));
}

#[test]
fn lists_content_types() {
    let dir = fixtures();
    let output = Command::new(env!("CARGO_BIN_EXE_magika"))
        .arg("--list-content-types")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1 + magika_burn::ContentType::ALL.len());
    // The header, the first rows and a few well-known content types.
    let selected: Vec<&str> = lines[..4]
        .iter()
        .chain(lines.iter().filter(|line| {
            ["python ", "rust ", "unknown "]
                .iter()
                .any(|label| line.starts_with(label))
        }))
        .copied()
        .collect();
    insta::assert_snapshot!(selected.join("\n"));
}

#[test]
fn lists_content_types_as_json() {
    let dir = fixtures();
    let output = Command::new(env!("CARGO_BIN_EXE_magika"))
        .args(["--list-content-types", "--json"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    let json: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json.len(), magika_burn::ContentType::ALL.len());
    let python = json.iter().find(|info| info["label"] == "python").unwrap();
    assert_eq!(
        *python,
        serde_json::json!({
            "label": "python",
            "mime_type": "text/x-python",
            "group": "code",
            "description": "Python source",
            "extensions": ["py", "pyi"],
            "is_text": true,
        })
    );
}

#[test]
fn rejects_invalid_usage() {
    let dir = fixtures();
    let mut output = String::new();
    for args in [
        &["-", "-"][..],
        &["-l", "-i", "main.rs"],
        &["-m", "sure", "main.rs"],
        &["--json", "-s", "main.rs"],
        &["--list-content-types", "main.rs"],
        &["--list-content-types", "--label"],
    ] {
        output += &magika(dir.path(), args, b"");
    }
    insta::assert_snapshot!(output);
}

#[test]
fn prints_help_without_arguments() {
    let dir = fixtures();
    insta::assert_snapshot!(magika(dir.path(), &[], b""));
}

#[test]
fn stops_quietly_when_stdout_is_closed() {
    let dir = fixtures();
    // Enough output to fill the pipe, so that writing fails once it is closed.
    let args = vec!["hello.txt"; 20_000];
    let mut child = Command::new(env!("CARGO_BIN_EXE_magika"))
        .args(&args)
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut first_line = String::new();
    stdout.read_line(&mut first_line).unwrap();
    drop(stdout);

    let output = child.wait_with_output().unwrap();
    assert_eq!(first_line, "hello.txt: Generic text document (text)\n");
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
}
