//! The book's catalogue of algorithms, generated from the repository's `ROADMAP.md` so that the
//! two can never disagree.
//!
//! The roadmap lists crates alphabetically, each as a `# Title (`crate`)` section with nested
//! `##`… subsections and `[x]` (implemented) or `[ ]` (planned) items. The catalogue regroups the
//! crates by subject, adds each crate's description (from its manifest) and links to its API
//! reference and, where one has been written, its book chapter.

use std::collections::HashMap;
use std::path::Path;

/// Subject groups, in reading order, and the crates in each.
const GROUPS: &[(&str, &str, &[&str])] = &[
    ("Calculus and analysis", "Derivatives, integrals, interpolation, equations and optimisation of functions, and the special functions of mathematical physics.", &["derivatives", "integration", "interpolation", "root_finding", "optimization", "special_functions"]),
    ("Linear algebra and transforms", "Dense matrix factorisations and eigenproblems, Fourier transforms, and readable reference implementations of both.", &["linalg", "linalg_reference", "ffts", "ffts_reference"]),
    ("Probability, statistics and learning", "Random numbers, statistical inference and machine learning.", &["random", "statistics", "machine_learning"]),
    ("Signals, information and fields", "Signal processing, information theory and coding, and electromagnetics.", &["dsp", "information", "electromagnetics"]),
    ("Discrete mathematics and geometry", "Number theory, graphs and computational geometry.", &["number_theory", "graph", "geometry"]),
    ("Foundations and input/output", "Shared types, utilities, sorting and file formats.", &["core", "utils", "sorting", "file_io"]),
];

/// One roadmap section: a crate (`level` 1) or a subsection of one.
#[derive(Default)]
struct Section {
    title: String,
    level: usize,
    lines: Vec<String>,
}

struct Crate {
    title: String,
    sections: Vec<Section>,
}

/// Splits the roadmap into crates (keyed by directory name) and the out-of-scope list.
fn parse(roadmap: &str) -> (HashMap<String, Crate>, Vec<String>) {
    let mut crates: HashMap<String, Crate> = HashMap::new();
    let mut out_of_scope = Vec::new();
    let mut current: Option<String> = None;
    let mut in_scope_list = false;
    for line in roadmap.lines() {
        let trimmed = line.trim();
        let level = trimmed.chars().take_while(|&c| c == '#').count();
        if level == 1 {
            in_scope_list = trimmed == "# Out of scope";
            current = None;
            if let (Some(open), Some(close)) = (trimmed.find("(`"), trimmed.rfind("`)")) {
                let key = trimmed[open + 2..close].to_string();
                let title = trimmed[1..open].trim().to_string();
                crates.insert(key.clone(), Crate { title, sections: vec![Section { level: 1, ..Default::default() }] });
                current = Some(key);
            }
            continue;
        }
        if in_scope_list {
            if let Some(item) = trimmed.strip_prefix("* ") {
                out_of_scope.push(item.to_string());
            }
            continue;
        }
        let Some(key) = &current else { continue };
        let entry = crates.get_mut(key).expect("current crate");
        if level > 1 {
            entry.sections.push(Section { title: trimmed[level..].trim().to_string(), level, lines: Vec::new() });
        } else if let Some(item) = trimmed.strip_prefix("[x] ") {
            entry.sections.last_mut().unwrap().lines.push(format!("- {item}"));
        } else if let Some(item) = trimmed.strip_prefix("[ ] ") {
            entry.sections.last_mut().unwrap().lines.push(format!("- {item} *(planned)*"));
        } else if !trimmed.is_empty() {
            entry.sections.last_mut().unwrap().lines.push(format!("\n{trimmed}\n"));
        }
    }
    (crates, out_of_scope)
}

/// The `description = "…"` of a crate's manifest.
fn description(root: &Path, key: &str) -> String {
    let manifest = std::fs::read_to_string(root.join("crates").join(key).join("Cargo.toml")).unwrap_or_default();
    manifest
        .lines()
        .find_map(|l| l.strip_prefix("description = \"").and_then(|r| r.strip_suffix('"')))
        .unwrap_or("")
        .to_string()
}

pub fn generate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let roadmap = std::fs::read_to_string(root.join("ROADMAP.md")).expect("read ROADMAP.md");
    let (crates, out_of_scope) = parse(&roadmap);
    let grouped: Vec<&str> = GROUPS.iter().flat_map(|g| g.2.iter().copied()).collect();
    for key in crates.keys() {
        assert!(grouped.contains(&key.as_str()), "crate `{key}` from ROADMAP.md is missing from the catalogue groups");
    }
    let count = |c: &Crate| c.sections.iter().flat_map(|s| &s.lines).filter(|l| l.starts_with("- ")).count();
    let total: usize = crates.values().map(count).sum();

    let mut md = String::new();
    md.push_str(&format!(
        "The library implements **{total} algorithms and components in {} crates**:\n\n",
        crates.len()
    ));
    md.push_str("| Subject | Crates | Items |\n|---|---|---:|\n");
    for (name, _, keys) in GROUPS {
        let anchor = name.to_lowercase().replace(", ", "-").replace(' ', "-").replace('/', "");
        let titles: Vec<String> = keys.iter().map(|k| crates[*k].title.clone()).collect();
        let items: usize = keys.iter().map(|k| count(&crates[*k])).sum();
        md.push_str(&format!("| [{name}](#{anchor}) | {} | {items} |\n", titles.join(", ")));
    }
    for (name, blurb, keys) in GROUPS {
        md.push_str(&format!("\n## {name}\n\n{blurb}\n"));
        for key in *keys {
            let c = &crates[*key];
            let package = format!("lib_jsl_{key}");
            md.push_str(&format!("\n### {}\n\n{}\n\n", c.title, description(&root, key)));
            let mut links = vec![format!("Crate `{package}`: [API reference](../api/{package}/index.html)")];
            if root.join("book/src").join(key).join("index.md").exists() {
                links.push(format!("[book chapter]({key}/index.md)"));
            }
            md.push_str(&format!("{} · {} items\n", links.join(" · "), count(c)));
            for s in &c.sections {
                if s.level > 1 {
                    md.push_str(&format!("\n{} {}\n", "#".repeat((s.level + 2).min(6)), s.title));
                }
                if !s.lines.is_empty() {
                    md.push('\n');
                    for l in &s.lines {
                        md.push_str(l);
                        md.push('\n');
                    }
                }
            }
        }
    }
    md.push_str("\n## What the library does not cover\n\n");
    for item in &out_of_scope {
        md.push_str(&format!("- {item}\n"));
    }
    crate::write("catalogue", "catalogue.md", &md);
}
