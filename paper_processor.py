"""Paper processor using Claude Agent SDK."""

import os
import shutil
import asyncio
from pathlib import Path
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from config import CLAUDE_ALLOWED_TOOLS, CLAUDE_PERMISSION_MODE, TEMPLATES_DIR


class PaperProcessor:
    """Process parsed papers using Claude Agent SDK."""
    
    def __init__(self):
        self.templates_dir = TEMPLATES_DIR
    
    async def generate_summaries(
        self,
        markdown_path: str,
        images_dir: str,
        output_dir: str,
        parser: str = "mineru"
    ) -> tuple[str, str, str]:
        """
        Generate executive summary, detailed breakdown, and relevant code analysis using Claude.
        
        Args:
            markdown_path: Path to the markdown file (full.md for MinerU, res.md for Zai)
            images_dir: Path to the images directory (images/ for MinerU, imgs/ for Zai)
            output_dir: Output directory for generated files
            parser: Parser type ('mineru' or 'zai')
            
        Returns:
            Tuple of (executive_summary_path, detailed_breakdown_path, relevant_code_path)
        """
        # Create output directory and images subdirectory
        os.makedirs(output_dir, exist_ok=True)
        output_images_dir = os.path.join(output_dir, "images")
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Create code_repo subdirectory for GitHub repositories
        code_repo_dir = os.path.join(output_dir, "code_repo")
        os.makedirs(code_repo_dir, exist_ok=True)
        
        # Copy images to output directory
        if os.path.exists(images_dir):
            print("Copying images to output directory...")
            for img_file in os.listdir(images_dir):
                src = os.path.join(images_dir, img_file)
                dst = os.path.join(output_images_dir, img_file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            print(f"Copied {len(os.listdir(output_images_dir))} images")
        else:
            print(f"Warning: Images directory not found at {images_dir}")
        
        # Prepare paths
        exec_summary_path = os.path.join(output_dir, "executive_summary.md")
        detailed_breakdown_path = os.path.join(output_dir, "detailed_breakdown.md")
        relevant_code_path = os.path.join(output_dir, "relevant_code.md")
        exec_template_path = os.path.join(self.templates_dir, "executive_summary.md")
        detailed_template_path = os.path.join(self.templates_dir, "detailed_breakdown.md")
        relevant_code_template_path = os.path.join(self.templates_dir, "relevant_code.md")
        
        # Build the prompt for Claude
        prompt = self._build_prompt(
            markdown_path,
            exec_template_path,
            detailed_template_path,
            relevant_code_template_path,
            exec_summary_path,
            detailed_breakdown_path,
            relevant_code_path,
            code_repo_dir,
            parser
        )
        
        print("\nStarting Claude Agent to generate summaries...")
        print("This may take a few minutes...\n")
        
        # Configure Claude SDK
        options = ClaudeAgentOptions(
            allowed_tools=CLAUDE_ALLOWED_TOOLS,
            permission_mode=CLAUDE_PERMISSION_MODE,
            cwd=output_dir
        )
        
        # Use Claude SDK to generate summaries
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            
            # Receive and process messages
            async for message in client.receive_response():
                # Just consume messages; Claude will write the files
                pass
        
        print("\nClaude Agent finished processing.")
        
        # Verify files were created
        if not os.path.exists(exec_summary_path):
            raise Exception("Executive summary was not created")
        if not os.path.exists(detailed_breakdown_path):
            raise Exception("Detailed breakdown was not created")
        if not os.path.exists(relevant_code_path):
            raise Exception("Relevant code documentation was not created")
        
        return exec_summary_path, detailed_breakdown_path, relevant_code_path
    
    def _build_prompt(
        self,
        markdown_path: str,
        exec_template_path: str,
        detailed_template_path: str,
        relevant_code_template_path: str,
        exec_output_path: str,
        detailed_output_path: str,
        relevant_code_output_path: str,
        code_repo_dir: str,
        parser: str
    ) -> str:
        """Build the prompt for Claude Agent."""
        
        # Get the markdown filename for display
        md_filename = os.path.basename(markdown_path)
        parser_name = "MinerU" if parser == "mineru" else "Zai"
        
        prompt = f"""You are a technical paper analyzer tasked with creating three comprehensive documents about a research paper: two summaries and a code analysis.

**Available Files:**
- Paper content: `{markdown_path}` (parsed by {parser_name})
- Executive summary template: `{exec_template_path}`
- Detailed breakdown template: `{detailed_template_path}`
- Relevant code template: `{relevant_code_template_path}`
- Images directory: `./images/` (contains figures from the paper)
- Code repository directory: `./code_repo/` (for cloning GitHub repositories)

**Your Task:**
Complete these steps in order:

**PART 1: Generate Paper Summaries**
1. Read the paper content from {markdown_path} carefully and thoroughly
2. Read the executive summary and detailed breakdown templates
3. Generate two markdown files following the templates exactly:
   - `{exec_output_path}`: A concise, engaging summary for non-technical readers
   - `{detailed_output_path}`: A comprehensive technical breakdown

**PART 2: Find and Analyze Implementation Code**
4. Use WebSearch to find the paper's GitHub repository:
   - Extract the paper title from the content
   - Search for: "[paper title] github" or "[paper title] code" or "[paper title] implementation"
   - Look for official implementations (often in paper or from authors)
   - Also try searching for the first author name + paper title + github
   - Check search results for github.com URLs

5. If GitHub repository found:
   - Use Bash to clone the repository: `cd ./code_repo && git clone <repo_url>`
   - Use Read, Glob, and Grep to explore the code structure
   - Identify key files that implement the paper's methods
   - Extract relevant code snippets

6. If NO GitHub repository found:
   - Generate illustrative pseudocode based on the paper's algorithms
   - Create conceptual code examples that demonstrate the key ideas
   - Base all pseudocode on the actual methods described in the paper

7. Read the relevant code template: `{relevant_code_template_path}`

8. Generate `{relevant_code_output_path}` following the template:
   - Document the repository info (or mark as "Pseudocode - No Repository Found")
   - Describe the architecture and key components
   - Include actual code snippets (if repo found) or pseudocode (if not found)
   - Map code back to specific sections of the paper
   - Provide usage examples

**Guidelines:**
- Follow all template structures precisely - keep all section headers and formatting
- Replace the placeholder text in brackets [...] with actual content from the paper
- Include relevant images using relative paths: `![caption](./images/filename.jpg)`
- Use Glob to see what images are available and reference them appropriately
- Include specific metrics, numbers, and results from the paper
- For the executive summary: Make it accessible and engaging for a general audience
- For the detailed breakdown: Provide technical depth while remaining clear
- For the relevant code: Provide practical implementation details
- Ensure all factual information comes from the paper content

**Process:**
1. Read the paper content and all three templates
2. Use Glob to see what images are available
3. Write the executive_summary.md file
4. Write the detailed_breakdown.md file
5. Use WebSearch to find the GitHub repository
6. If found: Clone with Bash and analyze the code; If not: Prepare pseudocode
7. Write the relevant_code.md file
8. Verify all three files are complete and properly formatted

Begin by reading the paper and templates, then generate all three documents."""

        return prompt

