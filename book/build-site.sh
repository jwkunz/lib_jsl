#!/usr/bin/env bash
# Builds the documentation site into target/site:
#   target/site/book/  the mdBook learning resource
#   target/site/api/   the rustdoc API reference
#   target/site/index.html  a landing page
# Requires mdbook and mdbook-katex. Pass --skip-figures to reuse the committed figures.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ "${1:-}" != "--skip-figures" ]]; then
    cargo run --quiet --release -p lib_jsl_book_generator
fi

RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D rustdoc::broken_intra_doc_links" cargo doc --quiet --workspace --no-deps --exclude lib_jsl_book_generator
cargo build --quiet --release --examples -p lib_jsl_integration

(cd book && mdbook build)

site="$root/target/site"
rm -rf "$site"
mkdir -p "$site"
cp -r "$root/target/book" "$site/book"
cp -r "$root/target/doc" "$site/api"
cp "$root/book/landing/index.html" "$site/index.html"
touch "$site/.nojekyll"
echo "Site written to $site"
