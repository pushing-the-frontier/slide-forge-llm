"""Agent prompt template for SlideForge."""

import json

AGENT_PROMPT_WITH_CONTENT = """## Presentation Brief

**Topic:** {topic}
**Target Audience:** {audience}
**Slides Required:** {num_slides}
**Sections per Slide:** {sections_per_slide}

## Source Data (USE THIS - DO NOT HALLUCINATE)

{content_data}

## Content Requirements

For this {audience} audience, ensure your content:
{audience_guidance}

## Current Progress
- Phase: {phase}
- Slides completed: {current_slide_count}/{num_slides}

## Available Tools
- `create_outline(sections)`: Create slide structure. sections=[{{"title": "...", "bullet_points": [...]}}]
- `generate_slide(slide_idx, title, sections)`: Create slide content. sections=[{{"heading": "...", "body": "..."}}]
- `edit_slide(slide_idx, title, sections)`: Revise existing slide
- `set_theme(theme_name)`: Set visual theme (corporate, tech, dark, creative, default)
- `review_deck()`: Check all slides for consistency and completeness
- `finalize()`: Complete the presentation

## Instructions
1. Create a comprehensive outline covering all {num_slides} slides using the source data above
2. Generate each slide using ONLY the provided data - do not make up numbers or facts
3. Review and refine before finalizing

Respond with exactly ONE JSON tool call wrapped in ```json ... ``` markers."""


AGENT_PROMPT_RESEARCH = """## Presentation Brief

**Topic:** {topic}
**Target Audience:** {audience}
**Slides Required:** {num_slides}
**Sections per Slide:** {sections_per_slide}

## Content Requirements

For this {audience} audience, ensure your content:
{audience_guidance}

## Current Progress
- Phase: {phase}
- Slides completed: {current_slide_count}/{num_slides}
- Research gathered: {research_count} items

## Available Tools
- `web_search(query)`: Search the web for current data, statistics, and facts. USE THIS FIRST!
- `create_outline(sections)`: Create slide structure. sections=[{{"title": "...", "bullet_points": [...]}}]
- `generate_slide(slide_idx, title, sections)`: Create slide content. sections=[{{"heading": "...", "body": "..."}}]
- `edit_slide(slide_idx, title, sections)`: Revise existing slide
- `set_theme(theme_name)`: Set visual theme (corporate, tech, dark, creative, default)
- `review_deck()`: Check all slides for consistency and completeness
- `finalize()`: Complete the presentation

## Instructions
1. FIRST: Use web_search 2-3 times to gather real data, statistics, and current information about the topic
2. Create a comprehensive outline covering all {num_slides} slides
3. Generate each slide using the researched data - include specific numbers and facts
4. Review and refine before finalizing

Respond with exactly ONE JSON tool call wrapped in ```json ... ``` markers."""


AGENT_PROMPT_EDIT = """## Presentation Edit Task

**Topic:** {topic}
**Target Audience:** {audience}
**Current Slides:** {current_slide_count}

## Edit Instructions

{edit_instructions}

## Existing Deck Summary

{deck_summary}

## Content Requirements

For this {audience} audience, ensure your content:
{audience_guidance}

## Available Tools

### Inspection
- `get_slide_content(slide_idx)`: Read the full content of a slide before editing
- `review_deck()`: Check all slides for consistency and completeness

### Content Editing
- `edit_slide(slide_idx, title, sections)`: Revise an existing slide's content. sections=[{{"heading": "...", "body": "..."}}]
- `set_theme(theme_name)`: Change visual theme (corporate, tech, dark, creative, default)

### Structural Editing
- `insert_slide(slide_idx, title, sections)`: Insert a new slide at position, shifting others down
- `delete_slide(slide_idx)`: Remove a slide from the deck
- `reorder_slides(from_idx, to_idx)`: Move a slide to a different position
- `duplicate_slide(slide_idx)`: Copy a slide (inserted right after original)

### Research & Creation
- `web_search(query)`: Search the web for data to improve slide content
- `generate_slide(slide_idx, title, sections)`: Overwrite/create a slide from scratch

### Finalization
- `finalize()`: Complete the editing session

## Instructions
1. FIRST: Use get_slide_content on a few key slides to understand the current deck
2. Identify issues: weak content, missing data, poor structure, ordering problems
3. Make targeted improvements using edit_slide, insert_slide, delete_slide, etc.
4. Use web_search if you need fresh data to strengthen any slide
5. Review with review_deck, then **call finalize() to complete the session**

IMPORTANT: You have a limited turn budget. Do NOT over-edit. Make focused, high-impact changes and call finalize() promptly. The session ends when you call finalize or run out of turns.

Respond with exactly ONE JSON tool call wrapped in ```json ... ``` markers."""


def _get_audience_guidance(audience: str) -> str:
    """Get specific guidance based on audience type."""
    audience_lower = audience.lower()
    
    if any(term in audience_lower for term in ['investor', 'vc', 'venture', 'angel', 'board']):
        return """- Lead with market opportunity and TAM/SAM/SOM
- Highlight traction metrics (ARR, MRR, growth rate, user counts)
- Show clear path to profitability or next funding milestone
- Include competitive differentiation and moat
- Present team credentials and relevant experience"""
    
    if any(term in audience_lower for term in ['cfo', 'finance', 'budget', 'treasury']):
        return """- Focus on ROI, payback period, and NPV analysis
- Include detailed cost breakdowns and projections
- Show variance analysis and budget impact
- Present risk-adjusted scenarios
- Use conservative assumptions with clear methodology"""
    
    if any(term in audience_lower for term in ['cto', 'engineering', 'technical', 'architect']):
        return """- Include architecture diagrams and technical specifications
- Address scalability, reliability, and performance metrics
- Show technology stack decisions and trade-offs
- Present implementation timeline and resource requirements
- Highlight technical risks and mitigation strategies"""
    
    if any(term in audience_lower for term in ['ceo', 'executive', 'c-suite', 'leadership']):
        return """- Lead with strategic impact and business outcomes
- Present high-level metrics with drill-down available
- Show alignment with company objectives and OKRs
- Include competitive context and market positioning
- Provide clear recommendations with decision points"""
    
    if any(term in audience_lower for term in ['sales', 'revenue', 'account']):
        return """- Focus on pipeline metrics and conversion rates
- Show quota attainment and forecast accuracy
- Include win/loss analysis and competitive insights
- Present territory and segment performance
- Highlight top deals and expansion opportunities"""
    
    if any(term in audience_lower for term in ['marketing', 'cmo', 'brand']):
        return """- Include CAC, LTV, and marketing ROI metrics
- Show channel performance and attribution data
- Present brand awareness and sentiment metrics
- Include campaign results and A/B test findings
- Highlight audience insights and segmentation"""
    
    if any(term in audience_lower for term in ['hr', 'people', 'talent', 'chro']):
        return """- Include headcount, attrition, and hiring metrics
- Show engagement scores and survey results
- Present compensation benchmarking data
- Include diversity and inclusion metrics
- Highlight training and development outcomes"""
    
    if any(term in audience_lower for term in ['product', 'roadmap']):
        return """- Focus on user metrics and engagement data
- Show feature adoption and usage analytics
- Present customer feedback and NPS trends
- Include competitive feature analysis
- Highlight prioritization framework and rationale"""
    
    # Default guidance
    return """- Use specific metrics and data points throughout
- Include industry benchmarks for context
- Present clear recommendations and next steps
- Tailor technical depth to audience expertise
- Focus on actionable insights over generic information"""


def _format_content_data(content: dict | None) -> str:
    """Format the content data for the prompt."""
    if not content:
        return "No source data provided. Use web_search to gather information."
    
    lines = []
    for key, value in content.items():
        key_display = key.replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"\n### {key_display}")
            for k, v in value.items():
                k_display = k.replace("_", " ").title()
                if isinstance(v, dict):
                    lines.append(f"**{k_display}:**")
                    for kk, vv in v.items():
                        lines.append(f"  - {kk}: {vv}")
                elif isinstance(v, list):
                    lines.append(f"**{k_display}:** {', '.join(str(x) for x in v)}")
                else:
                    lines.append(f"**{k_display}:** {v}")
        elif isinstance(value, list):
            lines.append(f"\n### {key_display}")
            for item in value:
                if isinstance(item, dict):
                    item_str = " | ".join(f"{k}: {v}" for k, v in item.items())
                    lines.append(f"- {item_str}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"**{key_display}:** {value}")
    
    return "\n".join(lines)


def _summarize_deck(state) -> str:
    """Build a text summary of existing slides for the edit prompt."""
    if not state.slides_html:
        return "(empty deck)"

    lines = []
    for i, html in enumerate(state.slides_html):
        if not html:
            lines.append(f"  Slide {i}: (empty)")
            continue
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one(".title")
        title = title_el.get_text(strip=True) if title_el else "(no title)"
        section_count = len(soup.select(".section"))
        word_count = len(soup.get_text(separator=" ", strip=True).split())
        lines.append(f"  Slide {i}: \"{title}\" — {section_count} sections, ~{word_count} words")
    return "\n".join(lines)


def format_prompt(state, content: dict | None = None) -> str:
    """Format the agent prompt with current state values."""
    brief = state.brief
    audience = brief.audience if brief else "general"

    if state.edit_mode:
        edit_instructions = (brief.edit_instructions if brief and brief.edit_instructions
                            else "Improve the overall quality, clarity, and impact of this presentation.")
        return AGENT_PROMPT_EDIT.format(
            topic=brief.topic if brief else "Unknown",
            audience=audience,
            current_slide_count=sum(1 for h in state.slides_html if h),
            edit_instructions=edit_instructions,
            deck_summary=_summarize_deck(state),
            audience_guidance=_get_audience_guidance(audience),
        )

    if content:
        return AGENT_PROMPT_WITH_CONTENT.format(
            topic=brief.topic if brief else "Unknown",
            audience=audience,
            num_slides=brief.num_slides if brief else 10,
            sections_per_slide=brief.sections_per_slide if brief else 3,
            content_data=_format_content_data(content),
            audience_guidance=_get_audience_guidance(audience),
            phase=state.phase,
            current_slide_count=sum(1 for h in state.slides_html if h),
        )
    else:
        return AGENT_PROMPT_RESEARCH.format(
            topic=brief.topic if brief else "Unknown",
            audience=audience,
            num_slides=brief.num_slides if brief else 10,
            sections_per_slide=brief.sections_per_slide if brief else 3,
            audience_guidance=_get_audience_guidance(audience),
            phase=state.phase,
            current_slide_count=sum(1 for h in state.slides_html if h),
            research_count=len(state.research_context),
        )
