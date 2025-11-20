"""
PDF generation for dialog transcripts.
"""
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF generation by removing markdown and fixing encoding issues."""
    if not text:
        return text

    # Remove markdown formatting
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_\n]+?)_(?!_)', r'\1', text)
    text = re.sub(r'\*\*+', '', text)
    text = re.sub(r'__+', '', text)
    text = re.sub(r'(?<=\w)\*(?=\w)', '', text)
    text = re.sub(r'(?<=\w)_(?=\w)', '', text)

    # Fix character encoding
    text = text.replace('\u2010', '-').replace('\u2011', '-').replace('\u2012', '-')
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2015', '-')
    text = text.replace('\u2212', '-').replace('\u25A0', ' ')
    text = text.replace('\u00A0', ' ').replace('\u2000', ' ').replace('\u2001', ' ')
    text = text.replace('\u2002', ' ').replace('\u2003', ' ').replace('\u2004', ' ')
    text = text.replace('\u2005', ' ').replace('\u2006', ' ').replace('\u2007', ' ')
    text = text.replace('\u2008', ' ').replace('\u2009', ' ').replace('\u200A', ' ')

    # Normalize spaces and line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\*+([^*\n]+?)\*+', r'\1', text)
    text = re.sub(r'_+([^_\n]+?)_+', r'\1', text)
    text = re.sub(r'\s+\*\s+', ' ', text)
    text = re.sub(r'\s+_\s+', ' ', text)
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\*+$', '', text, flags=re.MULTILINE)

    return text.strip()


def generate_pdf_from_dialog(dialog_data: Dict, prompt_config: Dict,
                             server_config: Dict, base_filename: str,
                             complete_dialog_data: Dict = None) -> Optional[str]:
    """Generate a PDF from dialog data with metadata, timing, and token counts.

    Args:
        dialog_data: Dialog conversation data
        prompt_config: Prompt configuration
        server_config: Server configuration with intermediator, participant1, participant2
        base_filename: Base filename for the PDF
        complete_dialog_data: Complete dialog data including GPU monitoring

    Returns:
        Path to generated PDF file, or None if failed
    """
    try:
        from utils import debug_log

        # Ensure output directory exists
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)

        pdf_path = output_dir / f"{base_filename}.pdf"

        # Extract server configs
        intermediator_config = server_config.get('intermediator', {})
        participant1_config = server_config.get('participant1', {})
        participant2_config = server_config.get('participant2', {})

        # Get dialog ID
        dialog_id = complete_dialog_data.get('dialog_id', 'unknown') if complete_dialog_data else 'unknown'

        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)

        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f4e78'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f4e78'),
            spaceAfter=8,
            spaceBefore=12
        )
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.leading = 14

        # Title
        elements.append(Paragraph("Intermediated Dialog Report", title_style))
        elements.append(Spacer(1, 0.2*inch))

        # Metadata Section
        elements.append(Paragraph("Metadata", heading_style))

        metadata_table_data = [
            ['Dialog ID:', dialog_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]

        if dialog_data.get('start_time') and dialog_data.get('end_time'):
            start_dt = datetime.fromtimestamp(dialog_data.get('start_time'))
            end_dt = datetime.fromtimestamp(dialog_data.get('end_time'))
            metadata_table_data.extend([
                ['Start Time:', start_dt.strftime('%Y-%m-%d %H:%M:%S')],
                ['End Time:', end_dt.strftime('%Y-%m-%d %H:%M:%S')],
                ['Duration:', f"{dialog_data.get('runtime_seconds', 0):.2f} seconds ({dialog_data.get('runtime_seconds', 0)/60:.2f} minutes)"],
            ])

        metadata_table_data.extend([
            ['Total Turns:', str(dialog_data.get('total_turns', 0))],
        ])

        metadata_table = Table(metadata_table_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e7f3ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.2*inch))

        # Participants Section
        elements.append(Paragraph("Participants", heading_style))

        participants_data = [
            ['Role', 'Name', 'Host', 'Model'],
            ['Intermediator', intermediator_config.get('name', 'N/A'),
             intermediator_config.get('host', 'N/A'), intermediator_config.get('model', 'N/A')],
            ['Participant 1', participant1_config.get('name', 'N/A'),
             participant1_config.get('host', 'N/A'), participant1_config.get('model', 'N/A')],
            ['Participant 2', participant2_config.get('name', 'N/A'),
             participant2_config.get('host', 'N/A'), participant2_config.get('model', 'N/A')],
        ]

        participants_table = Table(participants_data, colWidths=[1.5*inch, 2*inch, 2*inch, 2*inch])
        participants_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#dae3f3')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#e2efda')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fce4d6')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(participants_table)
        elements.append(Spacer(1, 0.2*inch))

        # Add GPU Energy data if available
        if complete_dialog_data:
            gpu_data = complete_dialog_data.get('gpu_data', {})
            if gpu_data and 'samples' in gpu_data:
                total_energy = 0.0
                samples = gpu_data.get('samples', [])
                server_energy = {'intermediator': 0.0, 'participant1': 0.0, 'participant2': 0.0}
                interval_hours = 1.0 / 3600.0

                for sample in samples:
                    servers = sample.get('servers', {})
                    for role in ['intermediator', 'participant1', 'participant2']:
                        server_data = servers.get(role, {})
                        gpus = server_data.get('gpus', [])
                        for gpu in gpus:
                            power_draw = gpu.get('power_draw_watts', 0)
                            if power_draw > 0:
                                server_energy[role] += power_draw * interval_hours

                total_energy = sum(server_energy.values())

                if total_energy > 0:
                    elements.append(Paragraph("Estimated Energy Consumption", heading_style))
                    energy_text = f"Total Energy: {total_energy:.4f} Wh<br/>"
                    for role, energy in server_energy.items():
                        if energy > 0:
                            role_name = role.replace('_', ' ').title()
                            energy_text += f"{role_name}: {energy:.4f} Wh<br/>"
                    elements.append(Paragraph(energy_text, normal_style))
                    elements.append(Spacer(1, 0.2*inch))

        # Prompt Configuration
        topic_prompt = prompt_config.get('intermediator_topic_prompt', '')
        if topic_prompt:
            elements.append(Paragraph("Topic / Instructions", heading_style))
            cleaned_topic = clean_text_for_pdf(topic_prompt)
            elements.append(Paragraph(cleaned_topic.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.1*inch))

        elements.append(PageBreak())

        # Conversation Section
        elements.append(Paragraph("Full Dialog", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        conversation_history = dialog_data.get('conversation_history', [])

        # Calculate totals
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in conversation_history:
            tokens = entry.get('tokens', {})
            if tokens:
                total_prompt_tokens += tokens.get('prompt_tokens', 0)
                total_completion_tokens += tokens.get('completion_tokens', 0)
                total_tokens += tokens.get('total', 0)

        # Write each turn
        for idx, entry in enumerate(conversation_history):
            turn = entry.get('turn', idx)
            speaker = entry.get('speaker', 'unknown')
            message = entry.get('message', '')
            is_summary = entry.get('is_summary', False)
            tokens = entry.get('tokens', {})

            # Format speaker name
            if speaker == 'intermediator':
                speaker_display = f"Moderator ({intermediator_config.get('name', 'Intermediator')})"
                bg_color = colors.HexColor('#dae3f3')
            elif speaker == 'participant1':
                speaker_display = f"Participant 1 ({participant1_config.get('name', 'Participant 1')})"
                bg_color = colors.HexColor('#e2efda')
            elif speaker == 'participant2':
                speaker_display = f"Participant 2 ({participant2_config.get('name', 'Participant 2')})"
                bg_color = colors.HexColor('#fce4d6')
            else:
                speaker_display = speaker.title()
                bg_color = colors.white

            # Turn header
            if is_summary:
                turn_label = "FINAL SUMMARY"
                header_style = ParagraphStyle(
                    'SummaryHeader',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor=colors.HexColor('#1f4e78'),
                    spaceAfter=8,
                    spaceBefore=12,
                    alignment=TA_CENTER
                )
                elements.append(Paragraph(turn_label, header_style))
            else:
                turn_label = f"Turn {turn}: {speaker_display}"
                header_style = ParagraphStyle(
                    'TurnHeader',
                    parent=styles['Heading3'],
                    fontSize=12,
                    textColor=colors.HexColor('#1f4e78'),
                    spaceAfter=6,
                    spaceBefore=10
                )
                elements.append(Paragraph(turn_label, header_style))

            elements.append(Spacer(1, 0.12*inch))

            # Message content
            message_style = ParagraphStyle(
                'Message',
                parent=normal_style,
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
                leftIndent=0.2*inch,
                rightIndent=0.2*inch,
                backColor=bg_color,
                borderPadding=10
            )
            cleaned_message = clean_text_for_pdf(message)
            message_escaped = cleaned_message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
            elements.append(Paragraph(message_escaped, message_style))
            elements.append(Spacer(1, 0.12*inch))

            # Token information
            if tokens and not is_summary:
                token_info_parts = []
                if tokens.get('prompt_tokens', 0) > 0:
                    token_info_parts.append(f"Prompt: {tokens.get('prompt_tokens', 0)}")
                if tokens.get('completion_tokens', 0) > 0:
                    token_info_parts.append(f"Completion: {tokens.get('completion_tokens', 0)}")
                if tokens.get('total', 0) > 0:
                    token_info_parts.append(f"Total: {tokens.get('total', 0)}")
                if tokens.get('tokens_per_second', 0) > 0:
                    token_info_parts.append(f"Speed: {tokens.get('tokens_per_second', 0):.2f} tokens/sec")
                if tokens.get('time_to_first_token', 0) > 0:
                    token_info_parts.append(f"TTFT: {tokens.get('time_to_first_token', 0):.3f}s")

                if token_info_parts:
                    token_text = " | ".join(token_info_parts)
                    token_style = ParagraphStyle(
                        'TokenInfo',
                        parent=normal_style,
                        fontSize=9,
                        textColor=colors.grey,
                        fontStyle='italic',
                        spaceBefore=4
                    )
                    elements.append(Paragraph(f"<i>Tokens: {token_text}</i>", token_style))

            elements.append(Spacer(1, 0.05*inch))

        # Summary statistics
        elements.append(PageBreak())
        elements.append(Paragraph("Summary Statistics", heading_style))

        stats_data = [
            ['Metric', 'Value'],
            ['Total Turns', str(len(conversation_history))],
            ['Total Prompt Tokens', f"{total_prompt_tokens:,}"],
            ['Total Completion Tokens', f"{total_completion_tokens:,}"],
            ['Total Tokens', f"{total_tokens:,}"],
        ]

        if dialog_data.get('runtime_seconds', 0) > 0 and total_tokens > 0:
            avg_tokens_per_second = total_tokens / dialog_data.get('runtime_seconds', 1)
            stats_data.append(['Average Tokens/Second', f"{avg_tokens_per_second:.2f}"])

        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ]))
        elements.append(stats_table)

        # Build PDF
        doc.build(elements)

        debug_log('info', f"PDF generated: {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        print(f"Failed to generate PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
