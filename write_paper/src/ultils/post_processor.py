import logging
import os
import re
import time
import json
import urllib.parse
import requests
import random
from bs4 import BeautifulSoup
from pathlib import Path
import shutil
from typing import Set, Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ========================
# json2bibtex functionality
# ========================

def extract_arxiv_ids_from_file(paper_folder: str, file_name: str) -> Set[str]:
    """
    Extract arXiv IDs from a TeX file.
    Function from json2bibtex/read_tex.py
    """
    with open(os.path.join(paper_folder, file_name), 'r', encoding='utf-8') as file:
        text = file.read()

    # Match \cite{} content
    pattern = r'\\cite\{([^}]*)\}'
    matches = re.findall(pattern, text)

    # Extract all references, which may contain multiple citations
    arxiv_ids = set()
    for match in matches:
        # Split multiple references, removing whitespace
        refs = re.split(r'[ ,]+', match)
        for ref in refs:
            if ref:
                arxiv_ids.add(ref)

    return arxiv_ids


def save_arxiv_ids_as_bib(arxiv_ids: Set[str], paper_folder: str, tex_file: str) -> None:
    """
    Convert arXiv IDs to BibTeX entries and save to file.
    Function from json2bibtex/trans_bibtex.py
    """
    # Output file path
    output_file = os.path.join(paper_folder, os.path.basename(tex_file).replace('.tex', '.bib'))

    # Load and parse arXiv metadata
    arxiv_metadata_path = os.path.join('json2bibtex', 'data', 'arxiv-metadata-oai-snapshot.json')

    bib_entries = []

    try:
        with open(arxiv_metadata_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, ncols=150, desc='bibtex'):
                metadata = json.loads(line.strip())

                # Check if current entry's ID is in our target list
                if metadata['id'] in arxiv_ids:
                    # Convert year format
                    year = int(metadata['id'][:2])
                    year = 2000 + year if year < 50 else 1900 + year

                    # Format authors
                    authors = ' and '.join([f"{author[1]} {author[0]}" for author in metadata['authors_parsed']])

                    # Create BibTeX entry
                    bib_entry = f"""@misc{{{metadata['id']},
  author = {{{authors}}},
  title = {{{metadata['title'].strip()}}},
  year = {{{str(year)}}},
  journal={{arXiv e-prints}},
  primaryClass = {{{metadata['categories']}}},
  url = {{https://arxiv.org/abs/{metadata['id']}}}
}}"""
                    bib_entries.append(bib_entry)
    except FileNotFoundError:
        logger.error(f"arXiv metadata file not found: {arxiv_metadata_path}")
        logger.warning("Creating empty BibTeX file")

    # Write BibTeX entries to file
    with open(output_file, 'w', encoding='utf-8') as bib_file:
        bib_file.write('\n\n'.join(bib_entries))

    logger.info(f"Saved BibTeX file: {output_file}")


def process_tex_files(paper_folder: str = './paper') -> None:
    """
    Process all TeX files in a folder to extract arXiv IDs and generate BibTeX entries.
    Based on json2bibtex/main.py
    """
    try:
        # Create paper directory if it doesn't exist
        if not os.path.exists(paper_folder):
            os.makedirs(paper_folder)
            logger.info(f"Created directory: {paper_folder}")

        # Get all TeX files in the paper folder
        tex_files = [f for f in os.listdir(paper_folder) if f.endswith('.tex')]

        if not tex_files:
            logger.warning(f"No TeX files found in {paper_folder}")
            return

        logger.info(f"Found {len(tex_files)} TeX files in {paper_folder}")

        # Process each TeX file
        for tex_file in tex_files:
            logger.info(f"Processing file: {tex_file}")

            # Extract arXiv IDs
            arxiv_ids = extract_arxiv_ids_from_file(paper_folder, tex_file)
            logger.info(f"Extracted {len(arxiv_ids)} arXiv IDs")

            # Save as BibTeX
            save_arxiv_ids_as_bib(arxiv_ids, paper_folder, tex_file)

    except Exception as e:
        logger.error(f"Error processing TeX files: {str(e)}")


# ========================
# LaTeX formatting functionality
# ========================

def convert_to_title_case(title: str) -> str:
    """
    Convert a string to title case with specific rules.
    Function from json2bibtex/省人力替换.py
    """
    # Check empty string
    if not title:
        return ""

    # Replace '&' with 'and'
    title = title.replace('&', 'and')

    # Words that should remain lowercase unless at start of title
    lowercase_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for',
                       'in', 'to', 'of', 'at', 'by', 'on'}

    words = title.split()
    result = []

    for i, word in enumerate(words):
        # Handle hyphenated words
        if '-' in word:
            hyphenated_parts = word.split('-')
            processed_parts = []

            for j, part in enumerate(hyphenated_parts):
                # Check if original case should be preserved
                if should_preserve_case(part):
                    processed_parts.append(part)
                # Capitalize part after hyphen
                elif j > 0:
                    processed_parts.append(part.capitalize())
                # First part follows normal rules
                elif i == 0 or part.lower() not in lowercase_words:
                    processed_parts.append(part.capitalize())
                else:
                    processed_parts.append(part.lower())

            result.append('-'.join(processed_parts))
        else:
            # Handle regular words
            if should_preserve_case(word):
                result.append(word)
            elif i == 0 or word.lower() not in lowercase_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())

    return ' '.join(result)


def should_preserve_case(word: str) -> bool:
    """
    Determine if a word's original case should be preserved.
    Function from json2bibtex/省人力替换.py
    """
    # Skip empty strings and single characters
    if not word or len(word) <= 1:
        return False

    # All uppercase (acronyms)
    if word.isupper():
        return True

    # Check for internal uppercase letters (like iPhone)
    if any(c.isupper() for c in word[1:]):
        return True

    return False


def modify_latex_file(file_name: str) -> None:
    """
    Apply formatting changes to a LaTeX file.
    Based on json2bibtex/省人力替换.py
    """
    logger.info(f"Formatting LaTeX file: {file_name}")

    try:
        # Read file
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        # Process filename to create title
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        title = base_name.replace('_', ' ')
        title = re.sub(r'\s+', ' ', title)
        title = title.replace('survey', '')
        title = convert_to_title_case(title)
        title = title + ": A Comprehensive Survey"

        # Check if formatting already applied
        if title in content and '\\usepackage{tikz}' in content:
            logger.info("Formatting already applied, skipping")
            return

        # Replacement 0: Remove cite package
        content = content.replace('\\usepackage{cite}\n', '')

        # Replacement 1: Add required packages
        old_text1 = '% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.'
        new_text1 = '''
\\usepackage{natbib}
\\usepackage{tikz}
\\usepackage{float}
\\usepackage{forest}
\\usetikzlibrary{arrows.meta, shapes.geometric, positioning, fit, backgrounds}
\\usepackage{booktabs}

\\usepackage{hyperref}  % Add hyperref for styling links
\\hypersetup{
    colorlinks=true,
    linkcolor=darkblue,  % Customize the color of the internal links (like citations)
    citecolor=darkblue,  % Customize the color of citations
    urlcolor=darkblue  % Customize the color of URLs
}
\\definecolor{darkblue}{rgb}{0.0, 0.0, 0.55}  % Define dark blue color
'''
        content = content.replace(old_text1, new_text1)

        # Replacement 2: Fix title
        old_text2 = '\\title{\n\\thanks{Identify applicable funding agency here. If none, delete this.}\n}'
        new_text2 = f'''\\title{{{title}
%
\\thanks{{\\hspace*{{-\\parindent}}\\rule{{3.8cm}}{{0.4pt}} \\\\
$\\ast$: Equal Contribution;  $\\dagger$: Corresponding Author.}}
}}'''

        # Try alternative formats if first attempt fails
        if old_text2 not in content:
            old_text2 = '\\title{\\thanks{Identify applicable funding agency here. If none, delete this.}}'
            if old_text2 not in content:
                old_text2 = '\\title{\\thanks{Identify applicable funding agency here. If none, delete this.}}'

        content = content.replace(old_text2, new_text2)

        # Add bibliography settings
        if '\\bibliographystyle{plain}' not in content:
            content = content.replace(
                '\\end{document}',
                f'''\\bibliographystyle{{apalike}}
\\bibliography{{{base_name}}}

\\end{{document}}'''
            )

        # Fix title spacing
        content = content.replace('\\title{ ', '\\title{')

        # Add spacing around section headers
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            if '\\section{' in lines[i]:
                # Insert two blank lines before section
                lines.insert(i, '')
                lines.insert(i, '')
                # Skip these newly inserted lines
                i += 2
                # Add newline after section header
                lines[i] = lines[i] + '\n'
            i += 1
        content = '\n'.join(lines)

        # Write modified content back to file
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)

        logger.info("LaTeX formatting completed successfully")

    except Exception as e:
        logger.error(f"Error formatting LaTeX file: {str(e)}")


def format_tex_files(folder_path: str = 'paper') -> None:
    """
    Format all TeX files in a folder.
    Based on json2bibtex/省人力替换.py
    """
    try:
        # Find all TeX files
        tex_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.tex'):
                    tex_files.append(os.path.join(root, file))

        if not tex_files:
            logger.warning(f"No TeX files found in {folder_path}")
            return

        logger.info(f"Found {len(tex_files)} TeX files to format")

        # Process each file
        for tex_file in tex_files:
            modify_latex_file(tex_file)

    except Exception as e:
        logger.error(f"Error formatting TeX files: {str(e)}")


# ========================
# cite_reverse-main functionality
# ========================

# Global variables for request management
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.29",
]
MIN_REQUEST_INTERVAL = 5  # Minimum wait time between requests
MAX_REQUEST_INTERVAL = 60  # Maximum wait time
CURRENT_INTERVAL = MIN_REQUEST_INTERVAL
last_request_time = 0


def make_request(url: str, timeout: int = 30):
    """
    Make HTTP request with rate limiting and user agent rotation.
    From cite_reverse-main/cite_reverse_dblp.py
    """
    global CURRENT_INTERVAL, last_request_time

    # Calculate wait time
    now = time.time()
    wait_time = max(0, last_request_time + CURRENT_INTERVAL + random.uniform(-1, 3) - now)

    if wait_time > 0:
        logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)

    # Randomize user agent
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Success - decrease interval slightly
        CURRENT_INTERVAL = max(MIN_REQUEST_INTERVAL, CURRENT_INTERVAL * 0.95)
        last_request_time = time.time()

        return response

    except requests.exceptions.HTTPError as e:
        # Increase interval if rate limited
        if e.response.status_code == 429 or "Too Many Requests" in str(e):
            CURRENT_INTERVAL = min(MAX_REQUEST_INTERVAL, CURRENT_INTERVAL * 1.5)
            logger.warning(f"Rate limit hit: increasing interval to {CURRENT_INTERVAL:.2f}s")
        last_request_time = time.time()
        raise


def parse_bib_file(file_path: str) -> List[str]:
    """
    Parse a BibTeX file and return list of entries.
    From cite_reverse-main/cite_reverse_dblp.py
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match complete BibTeX entries
    pattern = r"(@\w+\{[^@]+\})"
    entries = re.findall(pattern, content, re.DOTALL)
    return entries


def extract_entry_info(entry: str) -> Optional[Dict[str, str]]:
    """
    Extract metadata from a BibTeX entry.
    From cite_reverse-main/cite_reverse_dblp.py
    """
    # Extract entry type and cite key
    type_key_match = re.match(r"@(\w+)\{([^,]+),", entry)
    if not type_key_match:
        return None

    entry_type, cite_key = type_key_match.groups()

    # Extract author
    author_match = re.search(r"author\s*=\s*\{([^}]+)\}", entry)
    author = author_match.group(1) if author_match else ""

    # Extract title
    title_match = re.search(r"title\s*=\s*\{([^}]+)\}", entry)
    title = title_match.group(1) if title_match else ""
    # Clean multi-line titles
    title = re.sub(r"\s+", " ", title)

    # Extract URL
    url_match = re.search(r"url\s*=\s*\{([^}]+)\}", entry)
    url = url_match.group(1) if url_match else ""

    return {
        "entry_type": entry_type,
        "cite_key": cite_key,
        "author": author,
        "title": title,
        "url": url,
        "full_entry": entry,
    }


def search_dblp(author: str, title: str) -> Optional[str]:
    """
    Search DBLP for a paper and return BibTeX entry if found.
    From cite_reverse-main/cite_reverse_dblp.py
    """
    # Build search query
    query = f"{title} {author.split(' and ')[0]}"  # Use title and first author
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://dblp.org/search?q={encoded_query}"

    try:
        logger.info(f"Searching DBLP: {search_url}")
        logger.info(f"Current request interval: {CURRENT_INTERVAL:.2f}s")

        # Make request
        response = make_request(search_url)

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find search results
        result_items = soup.select(".publ-list .entry")
        if not result_items:
            logger.info("No search results found")
            return None

        logger.info(f"Found {len(result_items)} search results")

        # Look for BibTeX link in results
        for item in result_items:
            bibtex_link = item.select_one('nav.publ a[href*="bibtex"]')
            if bibtex_link:
                bibtex_url = bibtex_link["href"]
                # Ensure URL is complete
                if not bibtex_url.startswith("http"):
                    if bibtex_url.startswith("/"):
                        bibtex_url = f"https://dblp.org{bibtex_url}"
                    else:
                        bibtex_url = f"https://dblp.org/{bibtex_url}"

                logger.info(f"Getting BibTeX: {bibtex_url}")

                # Get BibTeX content
                bibtex_response = make_request(bibtex_url)
                bibtex_soup = BeautifulSoup(bibtex_response.text, "html.parser")
                bibtex_content = bibtex_soup.select_one("#bibtex-section pre")

                if bibtex_content:
                    return bibtex_content.text

        logger.info("No BibTeX content found")
        return None

    except Exception as e:
        logger.error(f"Error searching DBLP: {str(e)}")
        return None


def replace_entry_with_dblp(arxiv_entry: str, dblp_entry: str) -> str:
    """
    Replace arXiv entry with DBLP entry, preserving original cite key.
    From cite_reverse-main/cite_reverse_dblp.py
    """
    if not dblp_entry:
        return arxiv_entry

    # Extract arXiv cite key
    arxiv_info = extract_entry_info(arxiv_entry)
    if not arxiv_info:
        return arxiv_entry

    cite_key = arxiv_info["cite_key"]

    # Replace cite key in DBLP entry
    dblp_entry_new = re.sub(r"@(\w+)\{([^,]+),", f"@\\1{{{cite_key},", dblp_entry)

    return dblp_entry_new


def process_bibtex_entry(entry: str, output_file: str) -> None:
    """
    Process single BibTeX entry, searching for published version in DBLP.
    Based on cite_reverse-main/cite_reverse_dblp.py
    """
    info = extract_entry_info(entry)

    if not info:
        # Write unchanged if can't parse
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(entry + "\n\n")
        return

    logger.info(f"Processing entry: {info['cite_key']}")

    # Search DBLP
    dblp_entry = search_dblp(info["author"], info["title"])

    with open(output_file, "a", encoding="utf-8") as f_out:
        if dblp_entry:
            # Check if still arXiv entry in DBLP
            is_arxiv_in_dblp = re.search(
                r"eprinttype\s*=\s*\{\s*arXiv\s*\}", dblp_entry, re.IGNORECASE
            )

            if is_arxiv_in_dblp:
                logger.info("Found DBLP match, but still arXiv preprint")
                # Add comment explaining why keeping original
                comment = "% 在DBLP中找到匹配的条目，但该文献在DBLP上也是预印本，未正式发表，所以保留原条目。\n"
                f_out.write(comment + entry + "\n\n")
            else:
                logger.info("Found DBLP match: replacing entry")
                new_entry = replace_entry_with_dblp(entry, dblp_entry)
                f_out.write(new_entry + "\n\n")
        else:
            logger.info("No DBLP match found: keeping original")
            # Add warning comment
            comment = f"% WARNING: 未在DBLP中找到匹配的条目: {info['cite_key']}。建议人工重新检查。\n"
            f_out.write(comment + entry + "\n\n")


def process_bibtex_file(input_file: str, output_file: str) -> None:
    """
    Process all entries in a BibTeX file.
    Based on cite_reverse-main/cite_reverse_dblp.py
    """
    logger.info(f"Processing BibTeX file: {input_file} -> {output_file}")

    # Parse BibTeX file
    entries = parse_bib_file(input_file)
    logger.info(f"Found {len(entries)} citation entries")

    # Create/empty output file
    with open(output_file, "w", encoding="utf-8"):
        pass

    # Process each entry
    for i, entry in enumerate(entries):
        logger.info(f"Processing entry {i+1}/{len(entries)}")
        try:
            process_bibtex_entry(entry, output_file)
        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            # Add delay after error
            time.sleep(random.uniform(10, 20))

    logger.info(f"BibTeX processing completed: {output_file}")


# ========================
# Main post-processing function
# ========================

def remove_duplicate_bibtex_entries(bib_file: str) -> None:
    """
    Remove duplicate entries from a BibTeX file by checking cite keys.

    Args:
        bib_file: Path to the BibTeX file to clean
    """
    logger.info(f"Removing duplicate entries from {bib_file}")

    # Parse the file
    entries = parse_bib_file(bib_file)
    if not entries:
        logger.warning(f"No entries found in {bib_file}")
        return

    # Track seen cite keys
    seen_keys = set()
    unique_entries = []

    # Filter unique entries
    for entry in entries:
        info = extract_entry_info(entry)
        if not info:
            # Keep entries we can't parse
            unique_entries.append(entry)
            continue

        if info["cite_key"] not in seen_keys:
            seen_keys.add(info["cite_key"])
            unique_entries.append(entry)

    # Write unique entries back to file
    logger.info(f"Reduced from {len(entries)} to {len(unique_entries)} unique entries")
    with open(bib_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(unique_entries))

def post_process_paper(tex_file: str) -> bool:
    """
    Run complete post-processing workflow on a generated paper.

    Args:
        tex_file: Path to the generated TeX file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting post-processing for {tex_file}")

        # Process directly in the original directory - no need for central "paper" directory
        original_dir = os.path.dirname(tex_file)
        tex_filename = os.path.basename(tex_file)
        base_name = os.path.splitext(tex_filename)[0]

        logger.info(f"Processing paper in its original directory: {original_dir}")

        # Step 1: Extract arXiv IDs and generate BibTeX
        logger.info("Step 1: Generating BibTeX from arXiv IDs")

        # Extract arXiv IDs from the TeX file
        arxiv_ids = extract_arxiv_ids_from_file(original_dir, tex_filename)
        logger.info(f"Extracted {len(arxiv_ids)} arXiv IDs")

        # Generate BibTeX file directly in the original directory
        bib_file_path = os.path.join(original_dir, base_name + '.bib')
        save_arxiv_ids_as_bib(arxiv_ids, original_dir, tex_filename)
        logger.info(f"Generated BibTeX file: {bib_file_path}")

        # Step 2: Format the TeX file
        logger.info("Step 2: Formatting TeX file")
        modify_latex_file(tex_file)

        # Step 3: Convert arXiv citations to published versions if BibTeX file exists
        if os.path.exists(bib_file_path):
            logger.info("Step 3: Converting arXiv citations to published versions")

            # Create temporary file for DBLP processing
            temp_dblp_path = os.path.join(original_dir, base_name + '_temp_dblp.bib')

            # Process BibTeX file
            process_bibtex_file(bib_file_path, temp_dblp_path)

            # Remove duplicate entries
            remove_duplicate_bibtex_entries(temp_dblp_path)

            # Copy back to original file
            shutil.copy2(temp_dblp_path, bib_file_path)
            logger.info(f"Updated BibTeX file with improved citations: {bib_file_path}")

            # Clean up temporary file
            try:
                os.remove(temp_dblp_path)
                logger.info(f"Removed temporary file: {temp_dblp_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file: {e}")
        else:
            logger.warning(f"BibTeX file not found at {bib_file_path}, skipping citation conversion")

        logger.info(f"Post-processing completed successfully for {tex_file}")
        return True

    except Exception as e:
        logger.error(f"Error during post-processing: {str(e)}")
        return False

