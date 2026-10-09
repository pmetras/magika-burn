//! `magika`: detect the content type of files with Google's Magika model.

use std::{
    borrow::Cow,
    fs::{self, File},
    io::{self, Read, Write},
    num::NonZero,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use anyhow::Result;
use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};
use magika_burn::{
    ContentType, ContentTypeInfo, Magika, OverwriteReason, Prediction, PredictionMode,
};
use serde::Serialize;
use walkdir::WalkDir;

/// Detect the content type of files with Google's Magika model.
#[derive(Parser)]
#[command(
    name = "magika",
    version,
    arg_required_else_help = true,
    override_usage = "magika [OPTIONS] <PATH>...\n       magika --list-content-types [--json]"
)]
struct Args {
    /// Files or directories to identify. `-` reads standard input.
    #[arg(required_unless_present = "list_content_types", value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Identify the files inside directories instead of the directories themselves.
    #[arg(short, long)]
    recursive: bool,

    /// Identify symbolic links as such instead of following them.
    #[arg(long)]
    no_dereference: bool,

    /// How confident the model must be for its prediction to be used.
    #[arg(short = 'm', long, value_name = "MODE", value_enum, default_value_t = Mode::HighConfidence)]
    prediction_mode: Mode,

    /// Print the label (e.g. `rust`) instead of the description.
    #[arg(short, long, conflicts_with = "mime_type")]
    label: bool,

    /// Print the MIME type (e.g. `application/x-rust`) instead of the description.
    #[arg(short = 'i', long)]
    mime_type: bool,

    /// Append the score of the prediction.
    #[arg(short = 's', long)]
    output_score: bool,

    /// Print a JSON array with the prediction and content type of each path.
    #[arg(long, conflicts_with_all = ["label", "mime_type", "output_score"])]
    json: bool,

    /// Identify this many files at a time. Defaults to the number of processors.
    #[arg(short = 'j', long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    jobs: Option<u32>,

    /// List all content types and exit. Combine with --json for JSON.
    #[arg(
        long,
        conflicts_with_all = [
            "paths", "recursive", "no_dereference", "prediction_mode",
            "label", "mime_type", "output_score",
        ],
    )]
    list_content_types: bool,
}

/// Command-line names of [`PredictionMode`].
#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Per-content-type thresholds, e.g. 0.75 for Markdown, and 0.5 for the rest.
    HighConfidence,
    /// A threshold of 0.5 for every content type.
    MediumConfidence,
    /// No threshold: always use the model's prediction.
    BestGuess,
}

impl From<Mode> for PredictionMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::HighConfidence => Self::HighConfidence,
            Mode::MediumConfidence => Self::MediumConfidence,
            Mode::BestGuess => Self::BestGuess,
        }
    }
}

/// The result of identifying one path.
struct Entry {
    path: PathBuf,
    result: io::Result<Prediction>,
}

/// One unit of work, in output order.
enum Job {
    /// A path to identify, which any worker can do.
    Path(PathBuf),
    /// An entry that is already known: standard input, which only the main thread
    /// reads, or a walk error.
    Done(Entry),
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.paths.iter().filter(|path| is_stdin(path)).count() > 1 {
        Args::command()
            .error(
                ErrorKind::ArgumentConflict,
                "standard input (`-`) can only be read once",
            )
            .exit();
    }

    let result = if args.list_content_types {
        list_content_types(args.json).map(|()| true)
    } else {
        run(&args)
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        // The reader went away, e.g. `magika … | head`.
        Err(error) if is_broken_pipe(&error) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("magika: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Identifies and prints every path. Returns whether all of them were identified.
///
/// Identifying a file is a few milliseconds of arithmetic, so the files are spread
/// over a pool of threads. They are printed in the order they were given, not the
/// order they finish, which keeps the output of a run reproducible. The paths are
/// therefore collected before any of them is identified, so a recursive run holds
/// one `PathBuf` per file and prints nothing until the walk is over.
fn run(args: &Args) -> Result<bool> {
    let magika = Magika::new().with_prediction_mode(args.prediction_mode.into());
    let mut output = Output::new(args);

    // Standard input is read here: it is a single stream, and reading it from a
    // worker would race with the other workers for no gain.
    let jobs: Vec<Job> = args
        .paths
        .iter()
        .flat_map(|path| -> Box<dyn Iterator<Item = Job>> {
            if is_stdin(path) {
                let result = read_stdin().map(|content| magika.identify_bytes(&content));
                Box::new(std::iter::once(Job::Done(Entry {
                    path: path.clone(),
                    result,
                })))
            } else if args.recursive {
                Box::new(walk(path, args.no_dereference))
            } else {
                Box::new(std::iter::once(Job::Path(path.clone())))
            }
        })
        .collect();

    for entry in identify_all(&magika, jobs, args) {
        output.print(entry)?;
    }
    output.finish()?;
    Ok(output.all_identified)
}

/// Identifies every job, in parallel, and returns the entries in the order of the
/// jobs.
fn identify_all(magika: &Magika, jobs: Vec<Job>, args: &Args) -> Vec<Entry> {
    let threads = args.jobs.map_or_else(
        || thread::available_parallelism().map_or(1, NonZero::get),
        |jobs| jobs as usize,
    );
    // One thread is this one, and threads beyond the number of paths idle.
    let paths = jobs
        .iter()
        .filter(|job| matches!(job, Job::Path(_)))
        .count();
    if threads <= 1 || paths <= 1 {
        return jobs
            .into_iter()
            .map(|job| match job {
                Job::Path(path) => Entry {
                    result: identify_path(magika, &path, args.no_dereference),
                    path,
                },
                Job::Done(entry) => entry,
            })
            .collect();
    }

    // Each worker takes the next job that is a path, so that a slow file holds up
    // only itself, and writes the entry back where the job was.
    let entries: Vec<Mutex<Option<Entry>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        for _ in 0..threads.min(paths) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    let Job::Path(path) = job else { continue };
                    let entry = Entry {
                        path: path.clone(),
                        result: identify_path(magika, path, args.no_dereference),
                    };
                    *entries[index].lock().expect("no worker panics") = Some(entry);
                }
            });
        }
    });

    jobs.into_iter()
        .zip(entries)
        .map(|(job, entry)| match job {
            Job::Done(entry) => entry,
            Job::Path(_) => entry
                .into_inner()
                .expect("no worker panics")
                .expect("every path was identified"),
        })
        .collect()
}

/// Prints entries as they are identified: as text lines, or as the elements of
/// a JSON array.
struct Output<'a> {
    args: &'a Args,
    stdout: io::StdoutLock<'static>,
    printed: usize,
    all_identified: bool,
}

impl<'a> Output<'a> {
    fn new(args: &'a Args) -> Self {
        Self {
            args,
            stdout: io::stdout().lock(),
            printed: 0,
            all_identified: true,
        }
    }

    fn print(&mut self, entry: Entry) -> io::Result<()> {
        self.all_identified &= entry.result.is_ok();
        if self.args.json {
            let json = serde_json::to_string_pretty(&JsonEntry::from(&entry))
                .expect("entries serialize to JSON");
            let separator = if self.printed == 0 { "[\n" } else { ",\n" };
            // Indent the element; newlines inside JSON strings are escaped.
            write!(self.stdout, "{separator}  {}", json.replace('\n', "\n  "))?;
        } else {
            writeln!(self.stdout, "{}", text(&entry, self.args))?;
        }
        self.printed += 1;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        match (self.args.json, self.printed) {
            (false, _) => Ok(()),
            (true, 0) => writeln!(self.stdout, "[]"),
            (true, _) => writeln!(self.stdout, "\n]"),
        }
    }
}

/// An element of the `--json` output.
#[derive(Serialize)]
#[serde(untagged)]
enum JsonEntry<'a> {
    Identified {
        path: Cow<'a, str>,
        prediction: Prediction,
        /// Information about `prediction.output`.
        info: &'static ContentTypeInfo,
    },
    Failed {
        path: Cow<'a, str>,
        error: String,
    },
}

impl<'a> From<&'a Entry> for JsonEntry<'a> {
    fn from(entry: &'a Entry) -> Self {
        let path = entry.path.to_string_lossy();
        match &entry.result {
            Ok(prediction) => Self::Identified {
                path,
                prediction: *prediction,
                info: prediction.info(),
            },
            Err(error) => Self::Failed {
                path,
                error: error.to_string(),
            },
        }
    }
}

/// Prints every content type, as an aligned table or as a JSON array.
fn list_content_types(json: bool) -> Result<()> {
    let infos: Vec<&ContentTypeInfo> = ContentType::ALL.iter().map(|ct| ct.info()).collect();
    let mut stdout = io::stdout().lock();
    if json {
        let json = serde_json::to_string_pretty(&infos).expect("content types serialize to JSON");
        writeln!(stdout, "{json}")?;
        return Ok(());
    }

    let header = ["LABEL", "MIME TYPE", "GROUP", "EXTENSIONS", "DESCRIPTION"];
    let rows: Vec<[Cow<'_, str>; 5]> = infos
        .iter()
        .map(|info| {
            [
                info.label.into(),
                info.mime_type.into(),
                info.group.into(),
                info.extensions.join(",").into(),
                info.description.into(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let header = header.map(Cow::from);
    for row in std::iter::once(&header).chain(&rows) {
        let [label, mime_type, group, extensions, description] = row;
        writeln!(
            stdout,
            "{label:<0$}  {mime_type:<1$}  {group:<2$}  {extensions:<3$}  {description}",
            widths[0], widths[1], widths[2], widths[3],
        )?;
    }
    Ok(())
}

fn is_stdin(path: &Path) -> bool {
    path.as_os_str() == "-"
}

fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<io::Error>()
        .is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe)
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut content = Vec::new();
    io::stdin().lock().read_to_end(&mut content)?;
    Ok(content)
}

/// The files under `path` (or `path` itself if it is not a directory), depth-first
/// in name order. A walk error becomes a job holding it, so that it is reported in
/// the position where the path would have been.
fn walk<'a>(path: &'a Path, no_dereference: bool) -> impl Iterator<Item = Job> + 'a {
    WalkDir::new(path)
        .follow_links(!no_dereference)
        .follow_root_links(!no_dereference)
        .sort_by_file_name()
        .into_iter()
        .filter_map(move |entry| match entry {
            // Directories are walked, not identified.
            Ok(entry) if entry.file_type().is_dir() => None,
            Ok(entry) => Some(Job::Path(entry.into_path())),
            Err(error) => Some(Job::Done(walk_error(error, path))),
        })
}

/// Converts a walk error, whose message would repeat the path, to an entry.
fn walk_error(error: walkdir::Error, root: &Path) -> Entry {
    let path = error.path().unwrap_or(root).to_owned();
    let result = Err(match error.loop_ancestor() {
        Some(ancestor) => io::Error::other(format!(
            "directory cycle: links to its ancestor {}",
            ancestor.display()
        )),
        None => error
            .into_io_error()
            .unwrap_or_else(|| io::Error::other("unknown error")),
    });
    Entry { path, result }
}

fn identify_path(magika: &Magika, path: &Path, no_dereference: bool) -> io::Result<Prediction> {
    if no_dereference && fs::symlink_metadata(path)?.is_symlink() {
        return Ok(Prediction::ruled(ContentType::Symlink));
    }
    let metadata = fs::metadata(path)?;
    if metadata.is_dir() {
        return Ok(Prediction::ruled(ContentType::Directory));
    }
    if !metadata.is_file() {
        return Err(io::Error::other("not a regular file"));
    }
    magika.identify_reader(File::open(path)?)
}

/// Formats an entry as `<path>: <content type>`, followed by the score if requested.
fn text(entry: &Entry, args: &Args) -> String {
    let path = entry.path.display();
    let prediction = match &entry.result {
        Ok(prediction) => prediction,
        Err(error) => return format!("{path}: error: {error}"),
    };
    let content_type = describe(prediction.output, args);
    if prediction.overwrite_reason == OverwriteReason::LowConfidence {
        // The score is the model's score for its guess, so it goes with the guess.
        let guess = describe(prediction.dl, args);
        let score = percent(prediction.score);
        format!("{path}: {content_type} [low confidence: {guess}, {score}]")
    } else if args.output_score {
        format!("{path}: {content_type} {}", percent(prediction.score))
    } else {
        format!("{path}: {content_type}")
    }
}

/// The label, the MIME type, or the description and group of a content type.
fn describe(content_type: ContentType, args: &Args) -> String {
    let info = content_type.info();
    if args.label {
        info.label.to_owned()
    } else if args.mime_type {
        info.mime_type.to_owned()
    } else {
        format!("{} ({})", info.description, info.group)
    }
}

/// Formats a score as a whole percentage, rounded down like upstream Magika.
fn percent(score: f32) -> String {
    format!("{}%", (f64::from(score) * 100.0).floor())
}
