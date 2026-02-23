"""
PDF Report Generator - ICU Patient Medical Reports
Generates professional medical reports with time-series graphs
"""

import io
import logging
from datetime import datetime
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

logger = logging.getLogger(__name__)


class ICUReportGenerator:
    """
    Generate professional ICU patient reports with time-series visualizations
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Create custom paragraph styles for medical reports"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceBefore=12,
            spaceAfter=6,
            borderWidth=1,
            borderColor=colors.HexColor('#3b82f6'),
            borderPadding=(2, 2, 2, 10),
            leftIndent=10
        ))
        
        # Clinical text
        self.styles.add(ParagraphStyle(
            name='ClinicalText',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        # Alert text
        self.styles.add(ParagraphStyle(
            name='AlertText',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.red,
            spaceBefore=6,
            spaceAfter=6
        ))
    
    def generate_report(
        self,
        patient_id: str,
        patient_data: Dict,
        vitals_history: List[Dict],
        risk_history: List[Dict],
        ai_summary: Optional[str] = None,
        time_range_hours: int = 12
    ) -> io.BytesIO:
        """
        Generate comprehensive ICU patient report
        
        Args:
            patient_id: Patient identifier
            patient_data: Current patient state
            vitals_history: Time-series vital signs data
            risk_history: Time-series risk predictions
            ai_summary: AI-generated clinical summary (optional)
            time_range_hours: Hours of data to include
            
        Returns:
            BytesIO buffer containing PDF
        """
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Header
        story.extend(self._create_header(patient_id, time_range_hours))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Demographics
        story.extend(self._create_demographics_section(patient_id, patient_data))
        story.append(Spacer(1, 0.2*inch))
        
        # Current Vital Signs Table
        story.extend(self._create_current_vitals_section(patient_data))
        story.append(Spacer(1, 0.2*inch))
        
        # Vital Signs Trend Charts
        # Always include this section, even with limited data
        story.extend(self._create_vitals_chart_section(vitals_history))
        story.append(Spacer(1, 0.2*inch))
        
        # Risk Score Trend Chart
        # Always include this section, even with limited data
        story.extend(self._create_risk_chart_section(risk_history))
        story.append(Spacer(1, 0.2*inch))
        
        # Anomaly Events Timeline
        story.extend(self._create_anomaly_section(vitals_history))
        story.append(Spacer(1, 0.2*inch))
        
        # AI Clinical Summary
        if ai_summary:
            story.extend(self._create_ai_summary_section(ai_summary))
            story.append(Spacer(1, 0.2*inch))
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        logger.info(f"Generated PDF report for {patient_id}")
        return buffer
    
    def _create_header(self, patient_id: str, time_range_hours: int) -> List:
        """Create report header"""
        
        elements = []
        
        # Title
        title = Paragraph("ICU Patient Medical Report", self.styles['ReportTitle'])
        elements.append(title)
        
        # Report metadata
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata = f"""
        <para align=center>
        <b>Patient ID:</b> {patient_id} | 
        <b>Generated:</b> {report_time} | 
        <b>Time Range:</b> Last {time_range_hours} hours
        </para>
        """
        elements.append(Paragraph(metadata, self.styles['Normal']))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#3b82f6')))
        
        return elements
    
    def _create_demographics_section(self, patient_id: str, patient_data: Dict) -> List:
        """Create patient demographics section"""
        
        elements = []
        elements.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        
        # Extract data
        vitals = patient_data.get('vitals', {})
        bed_id = vitals.get('bed_id', 'Unknown')
        floor = vitals.get('floor_id', 'ICU-1')
        
        # Demographics table
        data = [
            ['Patient ID:', patient_id, 'Location:', f'{floor} - {bed_id}'],
            ['Report Type:', 'ICU Vital Signs Monitoring', 'Status:', 'Active'],
        ]
        
        table = Table(data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_current_vitals_section(self, patient_data: Dict) -> List:
        """Create current vital signs table"""
        
        elements = []
        elements.append(Paragraph("Current Vital Signs", self.styles['SectionHeader']))
        
        vitals = patient_data.get('vitals', {})
        prediction = patient_data.get('prediction', {})
        
        # Current vitals data
        data = [
            ['Parameter', 'Value', 'Status', 'Normal Range'],
            ['Heart Rate', f"{vitals.get('heart_rate', 0):.0f} bpm", 
             self._get_status(vitals.get('hr_anomaly', False)), '60-100 bpm'],
            ['Blood Pressure', f"{vitals.get('systolic_bp', 0):.0f}/{vitals.get('diastolic_bp', 0):.0f} mmHg",
             self._get_status(vitals.get('sbp_anomaly', False)), '90-140/60-90 mmHg'],
            ['SpO2', f"{vitals.get('spo2', 0):.0f}%",
             self._get_status(vitals.get('spo2_anomaly', False)), '>95%'],
            ['Respiratory Rate', f"{vitals.get('respiratory_rate', 0):.0f} br/min",
             'Normal', '12-20 br/min'],
            ['Temperature', f"{vitals.get('temperature', 0):.1f}°C",
             'Normal', '36.5-37.5°C'],
            ['Lactate', f"{vitals.get('lactate', 0):.2f} mmol/L",
             self._get_status(vitals.get('lactate_anomaly', False)), '<2.0 mmol/L'],
            ['Shock Index', f"{vitals.get('shock_index', 0):.2f}",
             self._get_status(vitals.get('shock_index_anomaly', False)), '<0.7'],
            ['Risk Score', f"{prediction.get('risk_score', 0):.1f}%",
             self._get_risk_status(prediction.get('risk_score', 0)), '<30% (Low Risk)'],
        ]
        
        table = Table(data, colWidths=[2*inch, 1.5*inch, 1.25*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_vitals_chart_section(self, vitals_history: List[Dict]) -> List:
        """Create vital signs trend charts"""
        
        elements = []
        elements.append(Paragraph("Vital Signs Trends", self.styles['SectionHeader']))
        
        # Check if we have sufficient data
        if not vitals_history or len(vitals_history) < 2:
            warning_text = """
            <para>
            <b>⚠ Insufficient Historical Data:</b> System needs at least 2-3 minutes of continuous 
            monitoring to generate trend graphs. Currently {count} data points available.
            </para>
            <para>
            <i>Please wait a few minutes and regenerate the report to see vital signs trends.</i>
            </para>
            """.format(count=len(vitals_history) if vitals_history else 0)
            elements.append(Paragraph(warning_text, self.styles['Normal']))
            return elements
        
        # Create matplotlib figure
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.suptitle('Last 3 Hours - Vital Signs Trends', fontsize=14, fontweight='bold')
        
        # Extract time and data
        timestamps = []
        hr_vals = []
        bp_sys = []
        bp_dia = []
        spo2_vals = []
        rr_vals = []
        temp_vals = []
        lactate_vals = []
        
        for entry in vitals_history[-100:]:  # Last 100 points
            try:
                ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                timestamps.append(ts)
                hr_vals.append(entry.get('heart_rate', 0))
                bp_sys.append(entry.get('systolic_bp', 0))
                bp_dia.append(entry.get('diastolic_bp', 0))
                spo2_vals.append(entry.get('spo2', 0))
                rr_vals.append(entry.get('respiratory_rate', 0))
                temp_vals.append(entry.get('temperature', 0))
                lactate_vals.append(entry.get('lactate', 0))
            except Exception as e:
                logger.warning(f"Error parsing vitals history entry: {e}")
                continue
        
        if not timestamps or len(timestamps) < 2:
            warning_text = """
            <para>
            <b>⚠ Data Format Issue:</b> Unable to parse sufficient historical entries. 
            Parsed {parsed} out of {total} entries.
            </para>
            """.format(parsed=len(timestamps), total=len(vitals_history))
            elements.append(Paragraph(warning_text, self.styles['Normal']))
            return elements
        
        # Heart Rate
        axes[0, 0].plot(timestamps, hr_vals, 'b-', linewidth=2)
        axes[0, 0].set_title('Heart Rate (bpm)', fontweight='bold')
        axes[0, 0].set_ylabel('bpm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Upper Limit')
        axes[0, 0].axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Lower Limit')
        
        # Blood Pressure
        axes[0, 1].plot(timestamps, bp_sys, 'r-', linewidth=2, label='Systolic')
        axes[0, 1].plot(timestamps, bp_dia, 'b-', linewidth=2, label='Diastolic')
        axes[0, 1].set_title('Blood Pressure (mmHg)', fontweight='bold')
        axes[0, 1].set_ylabel('mmHg')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # SpO2
        axes[1, 0].plot(timestamps, spo2_vals, 'g-', linewidth=2)
        axes[1, 0].set_title('SpO2 (%)', fontweight='bold')
        axes[1, 0].set_ylabel('%')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=95, color='r', linestyle='--', alpha=0.5, label='Critical Threshold')
        
        # Respiratory Rate
        axes[1, 1].plot(timestamps, rr_vals, 'purple', linewidth=2)
        axes[1, 1].set_title('Respiratory Rate (br/min)', fontweight='bold')
        axes[1, 1].set_ylabel('br/min')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Temperature
        axes[2, 0].plot(timestamps, temp_vals, 'orange', linewidth=2)
        axes[2, 0].set_title('Temperature (°C)', fontweight='bold')
        axes[2, 0].set_ylabel('°C')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=37.5, color='r', linestyle='--', alpha=0.5)
        
        # Lactate
        axes[2, 1].plot(timestamps, lactate_vals, 'brown', linewidth=2)
        axes[2, 1].set_title('Lactate (mmol/L)', fontweight='bold')
        axes[2, 1].set_ylabel('mmol/L')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Critical Level')
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8)
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        # Add to PDF
        img = Image(img_buffer, width=7*inch, height=5.6*inch)
        elements.append(img)
        
        return elements
    
    def _create_risk_chart_section(self, risk_history: List[Dict]) -> List:
        """Create risk score trend chart"""
        
        elements = []
        elements.append(Paragraph("Risk Score Trend", self.styles['SectionHeader']))
        
        # Check if we have data
        if not risk_history or len(risk_history) < 2:
            warning_text = """
            <para>
            <b>⚠ Insufficient Risk History:</b> System needs at least 2-3 minutes of risk predictions 
            to generate trend chart. Currently {count} data points available.
            </para>
            """.format(count=len(risk_history) if risk_history else 0)
            elements.append(Paragraph(warning_text, self.styles['Normal']))
            return elements
        
        # Extract data
        timestamps = []
        risk_scores = []
        
        for entry in risk_history[-100:]:
            try:
                ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                timestamps.append(ts)
                risk_scores.append(entry.get('risk_score', 0))
            except Exception as e:
                logger.warning(f"Error parsing risk history entry: {e}")
                continue
        
        if not timestamps or len(timestamps) < 2:
            warning_text = """
            <para>
            <b>⚠ Data Format Issue:</b> Unable to parse sufficient risk history entries.
            Parsed {parsed} out of {total} entries.
            </para>
            """.format(parsed=len(timestamps), total=len(risk_history))
            elements.append(Paragraph(warning_text, self.styles['Normal']))
            return elements
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(timestamps, risk_scores, 'r-', linewidth=2.5, label='Risk Score')
        ax.fill_between(timestamps, risk_scores, alpha=0.3, color='red')
        
        # Risk zones
        ax.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Low Risk Threshold')
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
        
        ax.set_title('Last 3 Hours - Patient Deterioration Risk Trend', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Risk Score (%)', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45, fontsize=8)
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        # Add to PDF
        img = Image(img_buffer, width=7*inch, height=2.4*inch)
        elements.append(img)
        
        return elements
    
    def _create_anomaly_section(self, vitals_history: List[Dict]) -> List:
        """Create anomaly events summary"""
        
        elements = []
        elements.append(Paragraph("Anomaly Events", self.styles['SectionHeader']))
        
        # Find anomalies
        anomalies = []
        for entry in vitals_history[-50:]:  # Last 50 entries
            if entry.get('anomaly_flag', False):
                anomalies.append(entry)
        
        if not anomalies:
            elements.append(Paragraph("✓ No anomalies detected in the monitored period", 
                                     self.styles['ClinicalText']))
            return elements
        
        # Anomaly table
        data = [['Time', 'Type', 'Values', 'Severity']]
        
        for anomaly in anomalies[-10:]:  # Last 10 anomalies
            try:
                ts = datetime.fromisoformat(anomaly['timestamp'].replace('Z', '+00:00'))
                time_str = ts.strftime('%H:%M:%S')
                
                anomaly_types = []
                if anomaly.get('hr_anomaly'): anomaly_types.append('HR')
                if anomaly.get('sbp_anomaly'): anomaly_types.append('BP')
                if anomaly.get('spo2_anomaly'): anomaly_types.append('SpO2')
                if anomaly.get('lactate_anomaly'): anomaly_types.append('Lactate')
                
                values = f"HR:{anomaly.get('heart_rate', 0):.0f} BP:{anomaly.get('systolic_bp', 0):.0f}"
                severity = '⚠ Warning'
                
                data.append([time_str, ', '.join(anomaly_types), values, severity])
            except:
                continue
        
        table = Table(data, colWidths=[1.5*inch, 2*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fee2e2')]),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_ai_summary_section(self, ai_summary: str) -> List:
        """Create AI-generated clinical summary section"""
        
        elements = []
        elements.append(Paragraph("AI Clinical Summary", self.styles['SectionHeader']))
        
        summary_text = f"""
        <para>
        <i>Generated by AI-powered medical analysis system using RAG (Retrieval-Augmented Generation) 
        with real-time patient data context.</i>
        </para>
        <para>
        {ai_summary}
        </para>
        """
        
        elements.append(Paragraph(summary_text, self.styles['ClinicalText']))
        return elements
    
    def _create_footer(self) -> List:
        """Create report footer"""
        
        elements = []
        elements.append(Spacer(1, 0.3*inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        
        footer_text = """
        <para align=center fontSize=8>
        <i>This report is generated by VitalX ICU Digital Twin System | 
        For medical professional use only | 
        Report generation time: {}</i>
        </para>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        return elements
    
    def _get_status(self, is_anomaly: bool) -> str:
        """Get status string for vital sign"""
        return '⚠ Abnormal' if is_anomaly else '✓ Normal'
    
    def _get_risk_status(self, risk_score: float) -> str:
        """Get risk status string"""
        if risk_score < 30:
            return '✓ Low Risk'
        elif risk_score < 70:
            return '⚠ Medium Risk'
        else:
            return '🚨 High Risk'
