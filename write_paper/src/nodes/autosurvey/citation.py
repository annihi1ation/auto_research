#!/usr/bin/env python3
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CitationManager:
    """Manages citations for the AutoSurvey pipeline"""

    def __init__(self):
        self._cited_papers = {}  # Dictionary to track papers cited in the content
        self._global_related_papers = []  # Global collection of all related papers

    def set_global_papers(self, papers):
        """Set the global collection of related papers"""
        self._global_related_papers = papers
        logger.info(f"Set {len(papers)} global papers for citation management")

    def extract_citations(self, sections: Dict[str, str]) -> None:
        """Extract citation IDs from all sections and track cited papers"""
        # Regular expressions to match both citation patterns: [Smith2019] and \cite{smith2019}
        old_citation_pattern = r'\[([^\]]+)\]'
        cite_pattern = r'\\cite\{([^}]+)\}'

        # Use the global collection of related papers
        reference_papers = self._global_related_papers

        logger.info(f"Processing citations from {len(reference_papers)} potential reference papers")

        for section_title, content in sections.items():
            if section_title not in ["Abstract", "title"]:
                # Find all \cite{} citations in the content
                cite_citations = re.findall(cite_pattern, content)

                # Also find traditional [AuthorYear] citations for backward compatibility
                old_citations = re.findall(old_citation_pattern, content)

                # Process all citations, prioritizing \cite{} format
                all_citations = cite_citations + old_citations

                logger.info(f"Found {len(all_citations)} citations in section: {section_title} ({len(cite_citations)} IEEE style, {len(old_citations)} traditional style)")

                for citation_id in all_citations:
                    # Skip numeric citations that might be footnotes or other references
                    if citation_id.isdigit():
                        continue

                    # If we haven't seen this citation before, add it to our tracking
                    if citation_id not in self._cited_papers:
                        # Try to find the paper in our reference collection
                        matched_paper = None
                        for paper in reference_papers:
                            # Check if author name is in the citation ID (common format: Smith2019)
                            author_match = False
                            if paper.authors:
                                first_author = paper.authors.split(',')[0].split(' ')[0] if ',' in paper.authors or ' ' in paper.authors else paper.authors
                                if first_author.lower() in citation_id.lower():
                                    author_match = True

                            # Check if year is in the citation ID
                            year_match = False
                            if paper.update_date and len(paper.update_date) >= 4:
                                year = paper.update_date[:4]
                                if year in citation_id:
                                    year_match = True

                            # If we have a good match, use this paper
                            if author_match or year_match:
                                matched_paper = paper
                                break

                        # If we found a matching paper, add it to our citations
                        if matched_paper:
                            self._cited_papers[citation_id] = matched_paper.to_dict()
                            logger.info(f"Found matching paper for citation: {citation_id}")
                        else:
                            # Create a fallback citation when no matching paper found
                            parts = citation_id.lower().split("cite{")[-1].split("}")
                            clean_key = parts[0] if parts and parts[0] else citation_id

                            # Extract potential author and year from the citation key
                            author_match = re.search(r'([a-z]+)', clean_key)
                            author = author_match.group(1) if author_match else "Unknown"

                            year_match = re.search(r'(\d{4})', clean_key)
                            year = year_match.group(1) if year_match else "2023"

                            self._cited_papers[citation_id] = {
                                'id': clean_key,
                                'title': f"Reference: {clean_key}",
                                'authors': f"{author.capitalize()} et al.",
                                'journal_ref': "Referenced in text",
                                'update_date': year
                            }
                            logger.warning(f"Created placeholder for citation with no matching paper: {citation_id}")

        logger.info(f"Extracted {len(self._cited_papers)} unique citations from the survey content")
        if not self._cited_papers:
            logger.warning("WARNING: No citations found in any section! The references.bib file will be empty.")
            # Add a default citation to ensure we have at least one reference
            self._cited_papers["default"] = {
                'id': "default",
                'title': "Example reference paper",
                'authors': "Author, Example",
                'journal_ref': "Journal of Important Research",
                'update_date': "2023"
            }

    def replace_citations_with_bibtex_format(self, content: str) -> str:
        """Replace citation IDs with BibTeX \cite{key} format and preserve existing \cite{} commands"""
        # First check if there are already \cite{} commands in the content
        cite_pattern = r'\\cite\{([^}]+)\}'
        existing_citations = re.findall(cite_pattern, content)

        if existing_citations:
            logger.info(f"Found {len(existing_citations)} existing \\cite{{}} citations")

        # Add existing citations to our tracking
        for citation_key in existing_citations:
            if citation_key not in self._cited_papers:
                # Try to find a paper that matches this citation key
                matched_paper = None
                for paper in self._global_related_papers:
                    author = ""
                    if paper.authors:
                        if "," in paper.authors:
                            author = paper.authors.split(",")[0].strip().lower()
                        else:
                            author = paper.authors.split(" ")[0].lower()
                        author = ''.join(c for c in author if c.isalnum())

                    year = ""
                    if paper.update_date and len(paper.update_date) >= 4:
                        year = paper.update_date[:4]

                    # Check if citation key contains author and year
                    if author and year and author.lower() in citation_key.lower() and year in citation_key:
                        matched_paper = paper
                        break

                # If we found a match, add it to our citations
                if matched_paper:
                    self._cited_papers[citation_key] = matched_paper.to_dict()
                    logger.info(f"Found matching paper for \\cite{{{citation_key}}}")
                else:
                    # Create a fallback citation
                    author_match = re.search(r'([a-z]+)', citation_key)
                    author = author_match.group(1) if author_match else "Unknown"

                    year_match = re.search(r'(\d{4})', citation_key)
                    year = year_match.group(1) if year_match else "2023"

                    self._cited_papers[citation_key] = {
                        'id': citation_key,
                        'title': f"Reference: {citation_key}",
                        'authors': f"{author.capitalize()} et al.",
                        'journal_ref': "Referenced in text",
                        'update_date': year
                    }
                    logger.warning(f"Created placeholder for \\cite{{{citation_key}}}")

        # If the content already uses \cite{} format throughout, return it as is
        if '[' not in content or ']' not in content:
            return content

        # Replace each [AuthorYear] citation with \cite{key} format (for backward compatibility)
        def replace_citation(match):
            citation_id = match.group(1)
            # If it's one of our tracked citations, replace it
            if citation_id in self._cited_papers:
                # Generate a stable citation key based on author and year
                citation_key = self.generate_citation_key(citation_id)
                return f"\\cite{{{citation_key}}}"
            # Otherwise, leave it as is
            return match.group(0)

        # Apply the replacements for [AuthorYear] format (keeping for backwards compatibility)
        citation_pattern = r'\[([^\]]+)\]'
        return re.sub(citation_pattern, replace_citation, content)

    def generate_citation_key(self, citation_id: str) -> str:
        """Generate a stable BibTeX citation key from a citation ID"""
        # If the citation_id already looks like a proper citation key, return it
        if citation_id.startswith('\\cite{') and citation_id.endswith('}'):
            return citation_id[6:-1]

        paper = self._cited_papers[citation_id]

        # Extract author
        if paper.get('authors'):
            if "," in paper['authors']:
                author = paper['authors'].split(",")[0].strip().lower()
            else:
                author = paper['authors'].split(" ")[0].lower()
            # Remove non-alphanumeric characters
            author = ''.join(c for c in author if c.isalnum())
        else:
            author = "unknown"

        # Extract year
        year = ""
        if paper.get('update_date') and len(paper.get('update_date')) >= 4:
            year = paper.get('update_date')[:4]
        else:
            year = "0000"

        # Create a unique identifier based on paper ID
        if paper.get('id') and paper.get('journal_ref'):
            # Use the first few characters of the paper ID
            unique_id = paper.get('id', '').replace('.', '').replace('-', '')[:4]
        else:
            # Use a hash of the title if there's no ID
            title_hash = hashlib.md5(paper.get('title', 'unknown').encode()).hexdigest()[:4]
            unique_id = title_hash

        return f"{author}{year}{unique_id}"

    def generate_bibtex_file(self, paper_dir: Path) -> str:
        """Generate a BibTeX file from the citation data and return the filename"""
        # Create BibTeX entries
        bibtex_entries = []

        logger.info(f"Generating BibTeX entries for {len(self._cited_papers)} citations")

        for citation_id, paper in self._cited_papers.items():
            # Generate a stable citation key
            citation_key = self.generate_citation_key(citation_id)

            # Format authors for BibTeX
            authors = paper.get('authors', "Unknown")

            # Format title
            title = paper.get('title', "Untitled")

            # Determine the type of publication
            if paper.get('journal_ref') and 'conference' in paper['journal_ref'].lower():
                entry_type = "inproceedings"
                venue_field = "booktitle"
                venue = paper['journal_ref']
            elif paper.get('journal_ref'):
                entry_type = "article"
                venue_field = "journal"
                venue = paper['journal_ref']
            else:
                entry_type = "misc"
                venue_field = "note"
                venue = f"arXiv:{paper.get('id', 'unknown')}"

            # Extract year
            year = ""
            if paper.get('update_date') and len(paper.get('update_date')) >= 4:
                year = paper.get('update_date')[:4]
            else:
                year = "2023"  # Default to current year if none available

            # Build the BibTeX entry
            entry = [
                f"@{entry_type}{{{citation_key},",
                f"  author = {{{authors}}},",
                f"  title = {{{title}}},",
                f"  {venue_field} = {{{venue}}},",
            ]

            if year:
                entry.append(f"  year = {{{year}}},")

            if paper.get('id') and entry_type != "misc":
                entry.append(f"  note = {{arXiv:{paper['id']}}},")

            entry.append("}")
            bibtex_entries.append("\n".join(entry))

        # If no citations, add a placeholder
        if not bibtex_entries:
            logger.warning("No citations found, adding placeholder reference")
            bibtex_entries.append("@article{placeholder,\n  author = {Placeholder, Author},\n  title = {This is a placeholder reference},\n  journal = {Journal of Examples},\n  year = {2023}\n}")

        # Write to file
        bibtex_filename = "references.bib"
        bibtex_path = paper_dir / bibtex_filename

        with open(bibtex_path, "w") as f:
            f.write("\n\n".join(bibtex_entries))

        logger.info(f"Generated BibTeX file with {len(bibtex_entries)} entries at {bibtex_path}")

        return bibtex_filename

    def format_paper(self, sections: Dict[str, str], paper_dir: Path, topic: str) -> Dict[str, str]:
        """Format the paper sections for output in IEEE format"""
        logger.info("DEBUG: Formatting paper in IEEE format")

        # Extract and track citations from all sections
        self.extract_citations(sections)
        logger.info(f"Tracked {len(self._cited_papers)} unique citations")

        logger.info(f"DEBUG: Formatting {len(sections)} sections: {', '.join(sorted(sections.keys()))}")

        # Generate the BibTeX file
        bibtex_filename = self.generate_bibtex_file(paper_dir)

        formatted_sections = {}

        # IEEE LaTeX preamble
        preamble = [
            "\\documentclass[conference]{IEEEtran}",
            "\\IEEEoverridecommandlockouts",
            "% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.",
            "\\usepackage{cite}",
            "\\usepackage{amsmath,amssymb,amsfonts}",
            "\\usepackage{algorithmic}",
            "\\usepackage{graphicx}",
            "\\usepackage{textcomp}",
            "\\usepackage{xcolor}",
            "\\def\\BibTeX{{\\rm B\\kern-.05em{\\sc i\\kern-.025em b}\\kern-.08em",
            "    T\\kern-.1667em\\lower.7ex\\hbox{E}\\kern-.125emX}}",
            "%",
            "% To compile this document with references:",
            "% 1. pdflatex filename.tex",
            "% 2. bibtex filename",
            "% 3. pdflatex filename.tex",
            "% 4. pdflatex filename.tex",
            "%",
            "\\begin{document}"
        ]
        formatted_sections["Preamble"] = "\n".join(preamble)

        # Title - Use the topic
        title_text = [
            f"\\title{{Research on {topic}}}",
            "\\author{\\IEEEauthorblockN{1\\textsuperscript{st} Given Name Surname}",
            "\\IEEEauthorblockA{\\textit{dept. name of organization (of Aff.)} \\\\",
            "\\textit{name of organization (of Aff.)}\\\\",
            "City, Country \\\\",
            "email address or ORCID}",
            "\\and",
            "\\IEEEauthorblockN{2\\textsuperscript{nd} Given Name Surname}",
            "\\IEEEauthorblockA{\\textit{dept. name of organization (of Aff.)} \\\\",
            "\\textit{name of organization (of Aff.)}\\\\",
            "City, Country \\\\",
            "email address or ORCID}}",
            "",
            "\\maketitle"
        ]
        formatted_sections["Title"] = "\n".join(title_text)

        # Abstract
        abstract = sections.get("Abstract", "")
        formatted_sections["Abstract"] = (
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
            "\\begin{IEEEkeywords}\n"
            "survey, literature review, comprehensive analysis, state-of-the-art\n"
            "\\end{IEEEkeywords}\n"
        )

        # Process main sections and replace citation placeholders with proper BibTeX citations
        for section_title, content in sections.items():
            if section_title not in ["Abstract", "title"]:
                # Replace citation placeholders [ID] with \cite{key} format
                processed_content = self.replace_citations_with_bibtex_format(content)

                # Format the section properly
                if "-" in section_title:  # This is a subsection
                    main_section, subsection = section_title.split(" - ", 1)
                    latex_content = f"\\subsection{{{subsection}}}\n{processed_content}"
                    formatted_sections[section_title] = latex_content
                else:
                    if section_title.lower() == "acknowledgment":
                        formatted_sections[section_title] = f"\\section*{{{section_title}}}\n{processed_content}"
                    else:
                        formatted_sections[section_title] = f"\\section{{{section_title}}}\n{processed_content}"

        # Use BibTeX for bibliography without file extension (LaTeX will add it)
        bibtex_name_without_extension = bibtex_filename.replace(".bib", "")
        formatted_sections["References"] = "\\bibliographystyle{IEEEtran}\n\\bibliography{" + bibtex_name_without_extension + "}"

        # Document end
        formatted_sections["End"] = "\\end{document}"

        # After processing references
        logger.info(f"DEBUG: Generated BibTeX file with {len(self._cited_papers)} citations")

        return formatted_sections
