# PDF Export Feature

## Overview
Added PDF generation functionality to the ICU Digital Twin UI. Users can now export patient reports and shift handoff documents as PDF files.

## Features Added

### 1. **Shift Handoff PDF Export**
- **Location**: Shift Handoff Modal (Nurse Dashboard)
- **Button**: Green "Export PDF" button in the modal footer
- **Contents**:
  - Shift summary (nurse, shift times, location, patient count)
  - Complete patient list with vitals
  - Critical vital sign alerts
  - Risk levels and trends
  - Recent interventions and alerts

### 2. **Patient Detail PDF Export**
- **Location**: Patient Detail Drawer (click any patient card)
- **Button**: "Export PDF" button in the header (next to close button)
- **Contents**:
  - Patient overview (name, bed, floor)
  - Current risk score and level
  - Complete vital signs table
  - Anomaly detection status

## How It Works

The PDF generation uses pure JavaScript (no external libraries required):
1. Creates a formatted HTML document with print-optimized CSS
2. Opens it in a new window
3. Triggers the browser's native print dialog
4. User can save as PDF using the "Save as PDF" option in the print dialog

## Usage Instructions

### For Shift Handoff:
1. Go to Nurse Dashboard
2. Click "View Shift Handoff" button
3. Click the green "Export PDF" button
4. In the print dialog, select "Save as PDF" as the destination
5. Choose location and save

### For Individual Patients:
1. Click on any patient card to open details
2. Click the "Export PDF" button in the header
3. In the print dialog, select "Save as PDF"
4. Save to desired location

## Files Modified

- ✅ `frontend/src/utils/pdfGenerator.ts` - PDF generation utility (NEW)
- ✅ `frontend/src/components/ShiftHandoffModal.tsx` - Added PDF export button
- ✅ `frontend/src/components/PatientDetailDrawer.tsx` - Added PDF export button

## Browser Compatibility

Works in all modern browsers:
- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ Requires popup permissions enabled

## Technical Details

**No dependencies added** - uses native browser APIs:
- `window.open()` for popup window
- `window.print()` for print dialog
- Pure CSS for formatting

**PDF Quality:**
- Professional medical report layout
- Color-coded risk indicators
- Responsive table formatting
- HIPAA confidentiality notice included
- Page-break optimization for printing

## Future Enhancements (Optional)

If you want advanced features later:
1. Add `jsPDF` + `html2canvas` for direct PDF generation (no print dialog)
2. Batch export multiple patient reports
3. Email integration
4. Digital signatures
5. Automated report scheduling

## Testing

To test:
1. Rebuild frontend: `docker compose build frontend`
2. Restart: `docker compose up -d frontend`
3. Open UI: http://localhost:3000
4. Test both export features

---

**Status**: ✅ Ready to use!
