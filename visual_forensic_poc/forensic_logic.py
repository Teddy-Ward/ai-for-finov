import os
import io
import json
import base64
import hashlib
from datetime import datetime
from PyPDF2 import PdfReader
from PIL import Image, ImageChops, ImageStat, ImageFilter, ImageDraw
import difflib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import fitz  # PyMuPDF - Python-only PDF renderer

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def pdf_to_images(pdf_bytes):
    """
    Convert PDF to images using PyMuPDF (fitz) - pure Python, no external dependencies.
    Returns list of PIL Images.
    """
    images = []
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Render page to image at 200 DPI
            mat = fitz.Matrix(200/72, 200/72)  # 200 DPI scaling
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        pdf_document.close()
        return images
    except Exception as e:
        # If PyMuPDF fails, create a blank image with error message
        print(f"PDF parsing error: {e}")
        error_img = Image.new('RGB', (1000, 1400), color='white')
        draw = ImageDraw.Draw(error_img)
        draw.text((50, 50), f"PDF Parsing Error:\n{str(e)}", fill='red')
        return [error_img]


# ============================================================================
# 1. DATA EXTRACTION & CROSS-VALIDATION (D2D)
# ============================================================================

def extract_structured_data_with_vision(image, document_type="payslip"):
    """
    Use LLM vision to extract structured data from document image.
    This replaces OCR with intelligent extraction that creates training data.
    """
    try:
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        extraction_prompt = f"""You are a document data extraction expert. Analyze this {document_type} image and extract ALL numerical and text data in a structured format.

For a UK PAYSLIP, extract:
- Employee details (Name, Employee ID, NI Number, Tax Code)
- Pay period dates
- Gross Pay
- All deductions (Tax, NI, Pension, etc.) with exact amounts
- Net Pay
- Year-to-date totals
- Bank details if visible
- Any other financial figures

For BANK STATEMENTS, extract:
- Account details
- All transactions with dates and amounts
- Opening and closing balances

Return a JSON object with:
1. "extracted_fields": All data fields found with their values
2. "financial_calculations": All monetary values and their labels
3. "confidence_scores": Your confidence (0-100%) for each extracted field
4. "anomalies_detected": Any unusual patterns, formatting issues, or inconsistencies you notice
5. "raw_text_regions": Key text regions and their content

Be extremely precise with numbers. Include currency symbols and decimal places exactly as shown."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data
        
    except Exception as e:
        return {
            "error": str(e),
            "extracted_fields": {},
            "financial_calculations": {},
            "confidence_scores": {},
            "anomalies_detected": [],
            "raw_text_regions": []
        }

def compare_extracted_data(gold_data, uploaded_data):
    """
    Cross-validate data between gold standard and uploaded document.
    Creates discrepancy report for training data.
    """
    discrepancies = []
    
    gold_fields = gold_data.get('extracted_fields', {})
    uploaded_fields = uploaded_data.get('extracted_fields', {})
    
    # Compare all fields
    all_keys = set(gold_fields.keys()) | set(uploaded_fields.keys())
    
    for key in all_keys:
        gold_value = gold_fields.get(key, "NOT PRESENT")
        uploaded_value = uploaded_fields.get(key, "NOT PRESENT")
        
        if str(gold_value).strip() != str(uploaded_value).strip():
            discrepancies.append({
                'field': key,
                'gold_standard': str(gold_value),
                'uploaded': str(uploaded_value),
                'severity': 'CRITICAL' if any(word in key.lower() for word in ['pay', 'tax', 'amount', 'balance']) else 'MODERATE'
            })
    
    return {
        'discrepancies': discrepancies,
        'total_fields_compared': len(all_keys),
        'match_rate': 1 - (len(discrepancies) / len(all_keys)) if all_keys else 1.0
    }

# ============================================================================
# 2. MATHEMATICAL & LOGICAL INTEGRITY
# ============================================================================

def validate_payslip_calculations(extracted_data):
    """
    Performs penny-perfect calculation validation.
    Verifies internal mathematical consistency.
    """
    validation_results = {
        'calculations_verified': [],
        'calculation_errors': [],
        'tax_ni_validation': {},
        'statistical_flags': []
    }
    
    try:
        financial_data = extracted_data.get('financial_calculations', {})
        
        # Extract key figures (handle various formats)
        def parse_currency(value):
            if isinstance(value, str):
                # Remove £, commas, and spaces
                cleaned = value.replace('£', '').replace(',', '').replace(' ', '')
                try:
                    return float(cleaned)
                except:
                    return None
            return float(value) if value else None
        
        # Validate: Gross - Deductions = Net
        gross = None
        net = None
        deductions = []
        
        for key, value in financial_data.items():
            key_lower = key.lower()
            if 'gross' in key_lower and 'pay' in key_lower:
                gross = parse_currency(value)
            elif 'net' in key_lower and 'pay' in key_lower:
                net = parse_currency(value)
            elif any(word in key_lower for word in ['tax', 'ni', 'deduction', 'pension', 'student loan']):
                amt = parse_currency(value)
                if amt:
                    deductions.append({'name': key, 'amount': amt})
        
        if gross and net and deductions:
            total_deductions = sum(d['amount'] for d in deductions)
            calculated_net = gross - total_deductions
            
            # Penny-perfect check
            if abs(calculated_net - net) < 0.01:
                validation_results['calculations_verified'].append({
                    'check': 'Gross - Deductions = Net',
                    'result': 'PASS',
                    'details': f'£{gross:.2f} - £{total_deductions:.2f} = £{net:.2f}'
                })
            else:
                validation_results['calculation_errors'].append({
                    'check': 'Gross - Deductions = Net',
                    'result': 'FAIL',
                    'expected': f'£{calculated_net:.2f}',
                    'actual': f'£{net:.2f}',
                    'difference': f'£{abs(calculated_net - net):.2f}',
                    'severity': 'CRITICAL'
                })
        
        # Statistical improbability checks
        if gross:
            # Check for suspicious rounding
            if gross == int(gross):
                validation_results['statistical_flags'].append({
                    'flag': 'Perfect Rounding',
                    'detail': f'Gross pay (£{gross:.2f}) is perfectly rounded - statistically rare',
                    'suspicion_level': 'MODERATE'
                })
            
            # Check for unusual patterns (e.g., all same digits)
            gross_str = f"{gross:.2f}".replace('.', '')
            if len(set(gross_str)) <= 2:
                validation_results['statistical_flags'].append({
                    'flag': 'Repetitive Digits',
                    'detail': f'Gross pay contains repetitive digit pattern',
                    'suspicion_level': 'HIGH'
                })
        
        # Tax code validation (basic check)
        extracted_fields = extracted_data.get('extracted_fields', {})
        tax_code = None
        ni_category = None
        
        for key, value in extracted_fields.items():
            if 'tax' in key.lower() and 'code' in key.lower():
                tax_code = value
            if 'ni' in key.lower() and 'category' in key.lower():
                ni_category = value
        
        validation_results['tax_ni_validation'] = {
            'tax_code': tax_code,
            'tax_code_format_valid': bool(tax_code and any(c.isdigit() for c in str(tax_code))),
            'ni_category': ni_category,
            'ni_category_valid': ni_category in ['A', 'B', 'C', 'H', 'J', 'M', 'Z'] if ni_category else False
        }
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results

# ============================================================================
# 3. VISUAL FORENSIC ANALYSIS
# ============================================================================

def spatial_fingerprinting(gold_image, uploaded_image):
    """
    Compares spatial layout at pixel level to detect template deviance.
    """
    try:
        # Ensure same size
        if gold_image.size != uploaded_image.size:
            uploaded_image = uploaded_image.resize(gold_image.size)
        
        # Convert to grayscale numpy arrays
        gold_gray = cv2.cvtColor(np.array(gold_image), cv2.COLOR_RGB2GRAY)
        upload_gray = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2GRAY)
        
        # Structural Similarity Index (SSIM)
        ssim_score, diff_img = ssim(gold_gray, upload_gray, full=True)
        diff_img = (diff_img * 255).astype("uint8")
        
        # Find regions with significant differences
        thresh = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        deviation_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Ignore tiny differences
                x, y, w, h = cv2.boundingRect(contour)
                deviation_regions.append({
                    'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'area_pixels': int(area),
                    'severity': 'HIGH' if area > 5000 else 'MODERATE'
                })
        
        return {
            'ssim_score': float(ssim_score),
            'layout_match': ssim_score > 0.95,
            'deviation_regions': deviation_regions[:10],  # Top 10 largest
            'total_deviations': len(deviation_regions)
        }
        
    except Exception as e:
        return {'error': str(e), 'ssim_score': 0, 'layout_match': False, 'deviation_regions': []}

def font_and_kerning_analysis(gold_image, uploaded_image):
    """
    Analyzes font consistency and character spacing patterns.
    Uses edge detection to identify font weight variations.
    """
    try:
        # Convert to grayscale
        gold_gray = cv2.cvtColor(np.array(gold_image.convert('L')), cv2.COLOR_GRAY2BGR)
        upload_gray = cv2.cvtColor(np.array(uploaded_image.convert('L')), cv2.COLOR_GRAY2BGR)
        
        # Edge detection to highlight text
        gold_edges = cv2.Canny(cv2.cvtColor(gold_gray, cv2.COLOR_BGR2GRAY), 100, 200)
        upload_edges = cv2.Canny(cv2.cvtColor(upload_gray, cv2.COLOR_BGR2GRAY), 100, 200)
        
        # Compare edge density (proxy for font weight)
        gold_edge_density = np.sum(gold_edges > 0) / gold_edges.size
        upload_edge_density = np.sum(upload_edges > 0) / upload_edges.size
        
        edge_difference = abs(gold_edge_density - upload_edge_density)
        
        # Detect font weight variations by region
        h, w = gold_edges.shape
        regions = []
        grid_size = 4
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                
                gold_region_density = np.sum(gold_edges[y1:y2, x1:x2] > 0) / ((y2-y1) * (x2-x1))
                upload_region_density = np.sum(upload_edges[y1:y2, x1:x2] > 0) / ((y2-y1) * (x2-x1))
                
                region_diff = abs(gold_region_density - upload_region_density)
                
                if region_diff > 0.02:  # Threshold for significant difference
                    regions.append({
                        'region': f'Grid_{i}_{j}',
                        'location': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                        'edge_density_diff': float(region_diff),
                        'suspicion': 'HIGH' if region_diff > 0.05 else 'MODERATE'
                    })
        
        return {
            'overall_edge_similarity': float(1 - edge_difference),
            'font_consistency_score': float(max(0, 1 - edge_difference * 10)),
            'suspicious_regions': regions,
            'analysis': 'Font weight variations detected' if edge_difference > 0.02 else 'Font consistency maintained'
        }
        
    except Exception as e:
        return {'error': str(e), 'overall_edge_similarity': 0, 'font_consistency_score': 0}

# ============================================================================
# 4. DEEP DIGITAL FORENSICS
# ============================================================================

def error_level_analysis(image):
    """
    Performs Error Level Analysis to detect image manipulation.
    Identifies areas with inconsistent compression levels.
    """
    try:
        # Save at known quality level
        temp_buffer = io.BytesIO()
        image.save(temp_buffer, 'JPEG', quality=90)
        temp_buffer.seek(0)
        
        # Reload and compare
        recompressed = Image.open(temp_buffer)
        
        # Calculate difference
        diff = ImageChops.difference(image.convert('RGB'), recompressed.convert('RGB'))
        
        # Enhance differences
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            scale = 1
        else:
            scale = 255.0 / max_diff
        
        ela_image = diff.point(lambda x: x * scale)
        
        # Analyze for suspicious regions
        ela_array = np.array(ela_image.convert('L'))
        suspicious_threshold = np.percentile(ela_array, 95)
        suspicious_regions = np.where(ela_array > suspicious_threshold)
        
        manipulation_score = len(suspicious_regions[0]) / ela_array.size
        
        return {
            'ela_performed': True,
            'manipulation_likelihood': float(manipulation_score),
            'risk_level': 'HIGH' if manipulation_score > 0.15 else 'MODERATE' if manipulation_score > 0.05 else 'LOW',
            'suspicious_pixel_percentage': float(manipulation_score * 100),
            'analysis': f'ELA detected {manipulation_score*100:.2f}% suspicious pixels'
        }
        
    except Exception as e:
        return {'error': str(e), 'ela_performed': False}

def metadata_xray(pdf_path):
    """
    Deep inspection of PDF metadata for editing software signatures.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            metadata = pdf_reader.metadata
            
        metadata_dict = dict(metadata) if metadata else {}
        
        # Check for suspicious software
        suspicious_software = ['photoshop', 'gimp', 'paint', 'editor', 'manipulator']
        editing_signatures = []
        
        for key, value in metadata_dict.items():
            value_str = str(value).lower()
            for software in suspicious_software:
                if software in value_str:
                    editing_signatures.append({
                        'field': key,
                        'signature': software,
                        'value': str(value),
                        'risk': 'CRITICAL'
                    })
        
        # Check creation/modification date consistency
        creation_date = metadata_dict.get('/CreationDate', '')
        mod_date = metadata_dict.get('/ModDate', '')
        
        return {
            'metadata': metadata_dict,
            'editing_signatures': editing_signatures,
            'has_suspicious_software': len(editing_signatures) > 0,
            'creation_date': creation_date,
            'modification_date': mod_date,
            'dates_match': creation_date == mod_date,
            'risk_assessment': 'CRITICAL' if editing_signatures else 'LOW'
        }
        
    except Exception as e:
        return {'error': str(e), 'has_suspicious_software': False}

def linguistic_hallucination_detection(extracted_data):
    """
    Detects bizarre or non-standard terminology common in AI-generated forgeries.
    """
    suspicious_terms = [
        'ai deduction', 'tax cash fonts', 'digital payment calculation',
        'automated fund', 'synthetic salary', 'generated amount',
        'virtual payment', 'algorithmic deduction', 'computed wage'
    ]
    
    detections = []
    
    # Check all text fields
    all_text = json.dumps(extracted_data).lower()
    
    for term in suspicious_terms:
        if term in all_text:
            detections.append({
                'term': term,
                'severity': 'CRITICAL',
                'description': 'AI-generated forgery terminology detected'
            })
    
    # Check for unusual formatting patterns
    extracted_fields = extracted_data.get('extracted_fields', {})
    for key, value in extracted_fields.items():
        value_str = str(value).lower()
        # Check for excessive technical jargon in simple fields
        if any(word in value_str for word in ['algorithm', 'synthetic', 'generated', 'computed']):
            if key.lower() in ['name', 'address', 'employer']:
                detections.append({
                    'term': value_str,
                    'field': key,
                    'severity': 'HIGH',
                    'description': 'Unusual technical terminology in standard field'
                })
    
    return {
        'hallucinations_detected': len(detections),
        'detections': detections,
        'risk_level': 'CRITICAL' if detections else 'LOW'
    }

# ============================================================================
# 5. STRATEGIC INTELLIGENCE & RISK SCORING
# ============================================================================

def calculate_comprehensive_risk_score(all_analysis_results):
    """
    Generates final trust score (0-100%) synthesizing all checks.
    """
    scores = {
        'data_integrity': 100,
        'mathematical_validity': 100,
        'visual_forensics': 100,
        'digital_forensics': 100,
        'linguistic_integrity': 100
    }
    
    weights = {
        'data_integrity': 0.25,
        'mathematical_validity': 0.25,
        'visual_forensics': 0.20,
        'digital_forensics': 0.20,
        'linguistic_integrity': 0.10
    }
    
    # Data integrity scoring
    data_comparison = all_analysis_results.get('data_comparison', {})
    match_rate = data_comparison.get('match_rate', 1.0)
    scores['data_integrity'] = match_rate * 100
    
    # Mathematical validity
    math_validation = all_analysis_results.get('mathematical_validation', {})
    calc_errors = len(math_validation.get('calculation_errors', []))
    if calc_errors > 0:
        scores['mathematical_validity'] = max(0, 100 - (calc_errors * 30))
    
    # Visual forensics
    spatial = all_analysis_results.get('spatial_fingerprinting', {})
    font_analysis = all_analysis_results.get('font_analysis', {})
    scores['visual_forensics'] = (
        spatial.get('ssim_score', 0) * 50 +
        font_analysis.get('font_consistency_score', 0) * 50
    )
    
    # Digital forensics
    ela = all_analysis_results.get('ela_analysis', {})
    metadata = all_analysis_results.get('metadata_analysis', {})
    ela_risk = ela.get('manipulation_likelihood', 0)
    metadata_risk = 1.0 if metadata.get('has_suspicious_software', False) else 0.0
    scores['digital_forensics'] = max(0, 100 - (ela_risk * 100 + metadata_risk * 50))
    
    # Linguistic integrity
    linguistic = all_analysis_results.get('linguistic_analysis', {})
    hallucinations = linguistic.get('hallucinations_detected', 0)
    scores['linguistic_integrity'] = max(0, 100 - (hallucinations * 40))
    
    # Calculate weighted final score
    final_score = sum(scores[key] * weights[key] for key in scores.keys())
    
    # Check if this might be a completely different template
    data_comparison = all_analysis_results.get('data_comparison', {})
    match_rate = data_comparison.get('match_rate', 1.0)
    visual_comparison = all_analysis_results.get('visual_comparison', {})
    visual_similarity = visual_comparison.get('visual_similarity', 100)
    
    # If both data match rate and visual similarity are very low, it's likely a different template
    is_different_template = (match_rate < 0.3 and visual_similarity < 50)
    
    if is_different_template:
        # This is a completely different format, not fraud
        return {
            'final_trust_score': 0,  # Can't assess trust for unknown template
            'component_scores': scores,
            'risk_level': 'UNKNOWN_TEMPLATE',
            'verdict': 'UNKNOWN_TEMPLATE',
            'template_detection': {
                'is_different_template': True,
                'data_match_rate': match_rate,
                'visual_similarity': visual_similarity,
                'recommendation': 'MANUAL_REVIEW_REQUIRED'
            },
            'recommendation': 'This document appears to be a completely different payslip format/template (not matching the gold standard). This does NOT indicate fraud - it may be from a different employer or payroll provider. MANUAL REVIEW REQUIRED: If legitimate, add this as a new gold standard template.'
        }
    
    # Determine risk level for matching templates
    if final_score >= 90:
        risk_level = "LOW"
        verdict = "AUTHENTIC"
    elif final_score >= 70:
        risk_level = "MODERATE"
        verdict = "SUSPICIOUS"
    elif final_score >= 50:
        risk_level = "HIGH"
        verdict = "LIKELY_TAMPERED"
    else:
        risk_level = "CRITICAL"
        verdict = "TAMPERED"
    
    return {
        'final_trust_score': round(final_score, 2),
        'component_scores': scores,
        'risk_level': risk_level,
        'verdict': verdict,
        'template_detection': {
            'is_different_template': False,
            'data_match_rate': match_rate,
            'visual_similarity': visual_similarity
        },
        'recommendation': get_recommendation(final_score, all_analysis_results)
    }

def get_recommendation(score, analysis):
    """Generate actionable recommendations based on analysis."""
    if score >= 90:
        return "Document appears authentic. Proceed with standard verification."
    elif score >= 70:
        return "Minor inconsistencies detected. Recommend manual review of flagged items."
    elif score >= 50:
        return "Significant anomalies detected. Require additional documentation and manual underwriter review."
    else:
        return "Critical tampering indicators present. REJECT document and request original copy."

# ============================================================================
# COMPREHENSIVE ANALYSIS ORCHESTRATOR
# ============================================================================

def analyze_document_authenticity_comprehensive(uploaded_file, gold_standard_path):
    """
    Master function orchestrating all forensic analysis modules.
    Returns comprehensive results for LLM analysis and training data.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_version': '2.0_comprehensive'
    }
    
    try:
        # Calculate hash of uploaded file
        uploaded_file.seek(0)
        uploaded_hash = hashlib.md5(uploaded_file.read()).hexdigest()
        
        # Calculate hash of gold standard
        with open(gold_standard_path, 'rb') as file:
            gold_hash = hashlib.md5(file.read()).hexdigest()
        
        # Check if files are identical
        if uploaded_hash == gold_hash:
            results['files_identical'] = True
            results['identity_check'] = {
                'uploaded_hash': uploaded_hash,
                'gold_hash': gold_hash,
                'verdict': 'IDENTICAL FILES - This is the authentic gold standard document'
            }
            # Still run analysis but flag it
        else:
            results['files_identical'] = False
            results['identity_check'] = {
                'uploaded_hash': uploaded_hash,
                'gold_hash': gold_hash,
                'verdict': 'Different files - proceeding with forensic analysis'
            }
        
        # Load gold standard
        with open(gold_standard_path, 'rb') as file:
            gold_reader = PdfReader(file)
            gold_text = [page.extract_text() for page in gold_reader.pages]
            
            # Try to convert to images using PyMuPDF (may fail for corrupted PDFs)
            file.seek(0)
            gold_bytes = file.read()
            try:
                gold_images = pdf_to_images(gold_bytes)
                images_available = True
            except Exception as e:
                print(f"Warning: Could not render gold standard PDF to images: {e}")
                gold_images = None
                images_available = False
        
        # Load uploaded document
        uploaded_file.seek(0)
        uploaded_reader = PdfReader(uploaded_file)
        uploaded_text = [page.extract_text() for page in uploaded_reader.pages]
        
        uploaded_file.seek(0)
        uploaded_bytes = uploaded_file.read()
        try:
            uploaded_images = pdf_to_images(uploaded_bytes)
        except Exception as e:
            print(f"Warning: Could not render uploaded PDF to images: {e}")
            uploaded_images = None
            images_available = False
        
        # If we can't render images, skip visual analysis
        if not images_available or gold_images is None or uploaded_images is None:
            results['analysis_mode'] = 'TEXT_ONLY (PDF rendering failed - corrupted or unsupported format)'
            results['gold_data_extraction'] = {'extracted_fields': {}, 'error': 'Image rendering failed'}
            results['uploaded_data_extraction'] = {'extracted_fields': {}, 'error': 'Image rendering failed'}
            results['data_comparison'] = {
                'discrepancies': [],
                'total_fields_compared': 0,
                'match_rate': 0,
                'error': 'Cannot compare - PDF rendering failed'
            }
            results['mathematical_validation'] = {
                'calculations_verified': [],
                'calculation_errors': [],
                'tax_ni_validation': {},
                'statistical_flags': [],
                'error': 'Cannot validate - no data extracted'
            }
            results['visual_comparison'] = {'error': 'PDF rendering failed'}
            results['ela_analysis'] = {'error': 'PDF rendering failed'}
            results['metadata_analysis'] = {'has_suspicious_software': False}
            results['linguistic_analysis'] = {'hallucination_indicators': []}
            results['final_trust_score'] = 0
            results['final_risk_score'] = 100
            results['verdict'] = 'ANALYSIS_FAILED'
            results['risk_level'] = 'CRITICAL - PDF Corrupted or Unsupported Format'
            results['llm_comprehensive_analysis'] = {
                'fraud_probability': 'N/A',
                'executive_summary': 'Analysis failed: The PDF file appears to be corrupted or in an unsupported format. Visual forensic analysis could not be completed.',
                'verdict': 'ANALYSIS_FAILED',
                'key_findings': ['PDF rendering failed', 'File may be corrupted', 'Visual analysis unavailable'],
                'risk_factors': ['Unable to extract visual data', 'PDF structure errors detected'],
                'confidence_level': 'NONE',
                'recommendation': 'Request a new copy of the document or verify file integrity'
            }
            return results
        
        # If files are identical, set perfect scores
        if results['files_identical']:
            results['gold_data_extraction'] = extract_structured_data_with_vision(
                gold_images[0], "payslip"
            )
            results['uploaded_data_extraction'] = results['gold_data_extraction'].copy()
            results['data_comparison'] = {
                'discrepancies': [],
                'total_fields_compared': len(results['gold_data_extraction'].get('extracted_fields', {})),
                'match_rate': 1.0
            }
            results['mathematical_validation'] = {
                'calculations_verified': [{'check': 'Identity Check', 'details': 'All calculations valid - identical file'}],
                'calculation_errors': [],
                'tax_ni_validation': {},
                'statistical_flags': [],
                'overall_status': 'PASS'
            }
            results['visual_comparison'] = {
                'visual_similarity': 100.0,
                'difference_percentage': 0.0,
                'identical': True
            }
            results['ela_analysis'] = {'suspicious_regions': [], 'overall_score': 100}
            results['metadata_analysis'] = {'has_suspicious_software': False}
            results['linguistic_analysis'] = {'hallucination_indicators': []}
            results['final_risk_score'] = 0
            results['overall_authenticity_score'] = 100
            results['final_trust_score'] = 100.0
            results['verdict'] = 'AUTHENTIC'
            results['risk_level'] = 'AUTHENTIC (Identical to Gold Standard)'
            results['llm_comprehensive_analysis'] = {
                'fraud_probability': 0,
                'executive_summary': 'This document is identical to the gold standard reference document. File hash verification confirms authenticity with 100% certainty.',
                'verdict': 'AUTHENTIC',
                'key_findings': ['Files are byte-for-byte identical', 'Hash verification passed', 'Perfect match to gold standard'],
                'risk_factors': [],
                'confidence_level': 'VERY_HIGH'
            }
            
            return results
        
        # MODULE 1: Data Extraction (LLM Vision)
        results['gold_data_extraction'] = extract_structured_data_with_vision(
            gold_images[0], "payslip"
        )

        results['uploaded_data_extraction'] = extract_structured_data_with_vision(
            uploaded_images[0], "payslip"
        )
        results['data_comparison'] = compare_extracted_data(
            results['gold_data_extraction'],
            results['uploaded_data_extraction']
        )
        
        # MODULE 2: Mathematical Validation
        results['mathematical_validation'] = validate_payslip_calculations(
            results['uploaded_data_extraction']
        )
        
        # MODULE 3: Visual Forensics
        results['spatial_fingerprinting'] = spatial_fingerprinting(
            gold_images[0], uploaded_images[0]
        )
        results['font_analysis'] = font_and_kerning_analysis(
            gold_images[0], uploaded_images[0]
        )
        
        # MODULE 4: Deep Digital Forensics
        results['ela_analysis'] = error_level_analysis(uploaded_images[0])
        
        # Save uploaded file temporarily for metadata analysis
        temp_path = "temp_upload.pdf"
        uploaded_file.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        results['metadata_analysis'] = metadata_xray(temp_path)
        os.remove(temp_path)
        
        results['linguistic_analysis'] = linguistic_hallucination_detection(
            results['uploaded_data_extraction']
        )
        
        # MODULE 5: Risk Scoring
        risk_score = calculate_comprehensive_risk_score(results)
        results.update(risk_score)
        
        # Generate LLM-powered comprehensive analysis
        results['llm_comprehensive_analysis'] = generate_llm_forensic_report(results)
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        print(f"Analysis error: {error_msg}")
        return {
            'error': error_msg,
            'final_trust_score': 0,
            'verdict': 'ERROR',
            'risk_level': 'CRITICAL - Analysis Failed',
            'llm_comprehensive_analysis': {
                'fraud_probability': 'N/A',
                'executive_summary': f'Analysis failed due to error: {error_msg}. The document may be corrupted or in an unsupported format.',
                'verdict': 'ERROR',
                'key_findings': [f'Analysis error: {error_msg}'],
                'risk_factors': ['Document parsing failed', 'Unable to complete forensic analysis'],
                'confidence_level': 'NONE'
            },
            'gold_data_extraction': {'extracted_fields': {}, 'error': error_msg},
            'uploaded_data_extraction': {'extracted_fields': {}, 'error': error_msg},
            'data_comparison': {'discrepancies': [], 'total_fields_compared': 0, 'match_rate': 0},
            'mathematical_validation': {'calculations_verified': [], 'calculation_errors': [], 'tax_ni_validation': {}, 'statistical_flags': []},
            'visual_comparison': {'visual_similarity': 0, 'difference_percentage': 100},
            'ela_analysis': {'suspicious_regions': []},
            'metadata_analysis': {'has_suspicious_software': False},
            'linguistic_analysis': {'hallucination_indicators': []}
        }

def generate_llm_forensic_report(analysis_results):
    """
    Use LLM to synthesize all forensic findings into comprehensive report.
    """
    try:
        prompt = f"""You are a senior forensic document examiner. Analyze the following comprehensive document forensic analysis and provide a detailed expert report.

ANALYSIS RESULTS:

**Data Extraction & Cross-Validation:**
- Gold Standard Fields: {len(analysis_results.get('gold_data_extraction', {}).get('extracted_fields', {}))}
- Uploaded Document Fields: {len(analysis_results.get('uploaded_data_extraction', {}).get('extracted_fields', {}))}
- Data Match Rate: {analysis_results.get('data_comparison', {}).get('match_rate', 0):.1%}
- Discrepancies: {len(analysis_results.get('data_comparison', {}).get('discrepancies', []))}

**Mathematical Validation:**
{json.dumps(analysis_results.get('mathematical_validation', {}), indent=2)}

**Visual Forensics:**
- SSIM Score: {analysis_results.get('spatial_fingerprinting', {}).get('ssim_score', 0):.4f}
- Layout Deviations: {analysis_results.get('spatial_fingerprinting', {}).get('total_deviations', 0)}
- Font Consistency: {analysis_results.get('font_analysis', {}).get('font_consistency_score', 0):.2%}

**Digital Forensics:**
- ELA Manipulation Likelihood: {analysis_results.get('ela_analysis', {}).get('manipulation_likelihood', 0):.2%}
- Suspicious Software: {analysis_results.get('metadata_analysis', {}).get('has_suspicious_software', False)}
- Linguistic Hallucinations: {analysis_results.get('linguistic_analysis', {}).get('hallucinations_detected', 0)}

**Risk Assessment:**
- Final Trust Score: {analysis_results.get('final_trust_score', 0):.2f}%
- Verdict: {analysis_results.get('verdict', 'UNKNOWN')}
- Risk Level: {analysis_results.get('risk_level', 'UNKNOWN')}

Based on this multi-layered forensic analysis, provide:

1. **EXECUTIVE SUMMARY**: 2-3 sentence verdict on document authenticity

2. **CRITICAL FINDINGS**: List the top 5 most significant findings (with severity: CRITICAL/HIGH/MODERATE/LOW)

3. **DATA DISCREPANCIES**: Specific field-level differences found between gold standard and uploaded document

4. **MATHEMATICAL INTEGRITY**: Assessment of calculation accuracy and statistical anomalies

5. **VISUAL & SPATIAL ANALYSIS**: Layout deviations, font inconsistencies, and spatial tampering indicators

6. **DIGITAL MANIPULATION INDICATORS**: ELA findings, metadata concerns, and technical red flags

7. **FRAUD PROBABILITY ASSESSMENT**: Estimate 0-100% likelihood this is a fraudulent document

8. **RECOMMENDED ACTIONS**: Specific next steps for underwriters/fraud team

9. **TRAINING DATA NOTES**: Key patterns this case demonstrates for model training purposes

Return as JSON with these keys: executive_summary, critical_findings (array of objects with 'finding' and 'severity'), data_discrepancies, mathematical_integrity, visual_spatial_analysis, digital_manipulation, fraud_probability, recommended_actions (array), training_notes"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a forensic document examiner with expertise in fraud detection, data analysis, and digital forensics. Provide detailed, professional analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {
            "error": f"LLM analysis failed: {str(e)}",
            "executive_summary": "Analysis unavailable due to error"
        }

def save_training_data_comprehensive(analysis_results, human_feedback, training_data_folder):
    """
    Enhanced training data saver with comprehensive forensic details.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forensic_analysis_{timestamp}.json"
        filepath = Path(training_data_folder) / filename
        
        # Create comprehensive training record
        training_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0_comprehensive",
                "document_type": "payslip"
            },
            "automated_analysis": {
                "data_extraction": {
                    "gold": analysis_results.get('gold_data_extraction', {}),
                    "uploaded": analysis_results.get('uploaded_data_extraction', {}),
                    "comparison": analysis_results.get('data_comparison', {})
                },
                "mathematical_validation": analysis_results.get('mathematical_validation', {}),
                "visual_forensics": {
                    "spatial": analysis_results.get('spatial_fingerprinting', {}),
                    "font_analysis": analysis_results.get('font_analysis', {})
                },
                "digital_forensics": {
                    "ela": analysis_results.get('ela_analysis', {}),
                    "metadata": analysis_results.get('metadata_analysis', {}),
                    "linguistic": analysis_results.get('linguistic_analysis', {})
                },
                "risk_assessment": {
                    "final_trust_score": analysis_results.get('final_trust_score', 0),
                    "component_scores": analysis_results.get('component_scores', {}),
                    "verdict": analysis_results.get('verdict', 'UNKNOWN'),
                    "risk_level": analysis_results.get('risk_level', 'UNKNOWN')
                },
                "llm_analysis": analysis_results.get('llm_comprehensive_analysis', {})
            },
            "human_expert_review": human_feedback,
            "training_value": {
                "data_quality": "high",
                "use_cases": ["fraud_detection", "document_verification", "pattern_learning"],
                "model_feedback_signals": {
                    "correct_prediction": human_feedback.get('verdict') == analysis_results.get('verdict'),
                    "human_override": human_feedback.get('trust_score') != analysis_results.get('final_trust_score'),
                    "key_learning_points": human_feedback.get('training_notes', [])
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return {"success": True, "filepath": str(filepath)}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_raw_analysis_output(analysis_results, training_data_folder, document_name=None):
    """
    Save raw analysis output automatically for every analysis run.
    This creates training data independent of human review.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = document_name.replace('.pdf', '').replace(' ', '_') if document_name else 'unknown'
        filename = f"raw_analysis_{doc_name}_{timestamp}.json"
        filepath = Path(training_data_folder) / filename
        
        # Create complete raw analysis record for model training
        training_record = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0_comprehensive",
                "document_name": document_name,
                "auto_saved": True,
                "requires_human_review": analysis_results.get('final_trust_score', 0) < 90
            },
            
            # Complete analysis results for training
            "analysis_outputs": {
                # Module 1: Data Extraction
                "data_extraction": {
                    "gold_standard": {
                        "extracted_fields": analysis_results.get('gold_data_extraction', {}).get('extracted_fields', {}),
                        "financial_calculations": analysis_results.get('gold_data_extraction', {}).get('financial_calculations', {}),
                        "confidence_scores": analysis_results.get('gold_data_extraction', {}).get('confidence_scores', {}),
                        "anomalies": analysis_results.get('gold_data_extraction', {}).get('anomalies_detected', [])
                    },
                    "uploaded_document": {
                        "extracted_fields": analysis_results.get('uploaded_data_extraction', {}).get('extracted_fields', {}),
                        "financial_calculations": analysis_results.get('uploaded_data_extraction', {}).get('financial_calculations', {}),
                        "confidence_scores": analysis_results.get('uploaded_data_extraction', {}).get('confidence_scores', {}),
                        "anomalies": analysis_results.get('uploaded_data_extraction', {}).get('anomalies_detected', [])
                    },
                    "cross_validation": {
                        "discrepancies": analysis_results.get('data_comparison', {}).get('discrepancies', []),
                        "match_rate": analysis_results.get('data_comparison', {}).get('match_rate', 0),
                        "total_fields_compared": analysis_results.get('data_comparison', {}).get('total_fields_compared', 0)
                    }
                },
                
                # Module 2: Mathematical Validation
                "mathematical_integrity": {
                    "calculations_verified": analysis_results.get('mathematical_validation', {}).get('calculations_verified', []),
                    "calculation_errors": analysis_results.get('mathematical_validation', {}).get('calculation_errors', []),
                    "tax_ni_validation": analysis_results.get('mathematical_validation', {}).get('tax_ni_validation', {}),
                    "statistical_flags": analysis_results.get('mathematical_validation', {}).get('statistical_flags', [])
                },
                
                # Module 3: Visual Forensics
                "visual_forensics": {
                    "spatial_fingerprinting": {
                        "ssim_score": analysis_results.get('spatial_fingerprinting', {}).get('ssim_score', 0),
                        "layout_match": analysis_results.get('spatial_fingerprinting', {}).get('layout_match', False),
                        "deviation_regions": analysis_results.get('spatial_fingerprinting', {}).get('deviation_regions', []),
                        "total_deviations": analysis_results.get('spatial_fingerprinting', {}).get('total_deviations', 0)
                    },
                    "font_analysis": {
                        "edge_similarity": analysis_results.get('font_analysis', {}).get('overall_edge_similarity', 0),
                        "consistency_score": analysis_results.get('font_analysis', {}).get('font_consistency_score', 0),
                        "suspicious_regions": analysis_results.get('font_analysis', {}).get('suspicious_regions', []),
                        "analysis_summary": analysis_results.get('font_analysis', {}).get('analysis', '')
                    }
                },
                
                # Module 4: Digital Forensics
                "digital_forensics": {
                    "error_level_analysis": {
                        "performed": analysis_results.get('ela_analysis', {}).get('ela_performed', False),
                        "manipulation_likelihood": analysis_results.get('ela_analysis', {}).get('manipulation_likelihood', 0),
                        "risk_level": analysis_results.get('ela_analysis', {}).get('risk_level', 'UNKNOWN'),
                        "suspicious_pixels_pct": analysis_results.get('ela_analysis', {}).get('suspicious_pixel_percentage', 0)
                    },
                    "metadata_analysis": {
                        "has_suspicious_software": analysis_results.get('metadata_analysis', {}).get('has_suspicious_software', False),
                        "editing_signatures": analysis_results.get('metadata_analysis', {}).get('editing_signatures', []),
                        "dates_match": analysis_results.get('metadata_analysis', {}).get('dates_match', True),
                        "risk_assessment": analysis_results.get('metadata_analysis', {}).get('risk_assessment', 'LOW')
                    },
                    "linguistic_analysis": {
                        "hallucinations_detected": analysis_results.get('linguistic_analysis', {}).get('hallucinations_detected', 0),
                        "detections": analysis_results.get('linguistic_analysis', {}).get('detections', []),
                        "risk_level": analysis_results.get('linguistic_analysis', {}).get('risk_level', 'LOW')
                    }
                },
                
                # Module 5: Risk Assessment & LLM Analysis
                "risk_assessment": {
                    "final_trust_score": analysis_results.get('final_trust_score', 0),
                    "component_scores": analysis_results.get('component_scores', {}),
                    "verdict": analysis_results.get('verdict', 'UNKNOWN'),
                    "risk_level": analysis_results.get('risk_level', 'UNKNOWN'),
                    "recommendation": analysis_results.get('recommendation', '')
                },
                
                "llm_comprehensive_analysis": {
                    "executive_summary": analysis_results.get('llm_comprehensive_analysis', {}).get('executive_summary', ''),
                    "critical_findings": analysis_results.get('llm_comprehensive_analysis', {}).get('critical_findings', []),
                    "data_discrepancies": analysis_results.get('llm_comprehensive_analysis', {}).get('data_discrepancies', ''),
                    "mathematical_integrity": analysis_results.get('llm_comprehensive_analysis', {}).get('mathematical_integrity', ''),
                    "visual_spatial_analysis": analysis_results.get('llm_comprehensive_analysis', {}).get('visual_spatial_analysis', ''),
                    "digital_manipulation": analysis_results.get('llm_comprehensive_analysis', {}).get('digital_manipulation', ''),
                    "fraud_probability": analysis_results.get('llm_comprehensive_analysis', {}).get('fraud_probability', ''),
                    "recommended_actions": analysis_results.get('llm_comprehensive_analysis', {}).get('recommended_actions', []),
                    "training_notes": analysis_results.get('llm_comprehensive_analysis', {}).get('training_notes', '')
                }
            },
            
            # Training labels and signals
            "training_labels": {
                "is_tampered": analysis_results.get('verdict', 'UNKNOWN') in ['TAMPERED', 'LIKELY_TAMPERED'],
                "is_authentic": analysis_results.get('verdict', 'UNKNOWN') == 'AUTHENTIC',
                "requires_manual_review": analysis_results.get('risk_level', 'UNKNOWN') in ['HIGH', 'CRITICAL'],
                "confidence_band": 'high' if analysis_results.get('final_trust_score', 0) >= 90 else 'medium' if analysis_results.get('final_trust_score', 0) >= 70 else 'low'
            },
            
            # Feature vectors for ML model training
            "feature_vectors": {
                "data_match_rate": analysis_results.get('data_comparison', {}).get('match_rate', 0),
                "ssim_score": analysis_results.get('spatial_fingerprinting', {}).get('ssim_score', 0),
                "font_consistency": analysis_results.get('font_analysis', {}).get('font_consistency_score', 0),
                "ela_manipulation_score": analysis_results.get('ela_analysis', {}).get('manipulation_likelihood', 0),
                "has_calculation_errors": len(analysis_results.get('mathematical_validation', {}).get('calculation_errors', [])) > 0,
                "has_metadata_issues": analysis_results.get('metadata_analysis', {}).get('has_suspicious_software', False),
                "discrepancy_count": len(analysis_results.get('data_comparison', {}).get('discrepancies', [])),
                "statistical_anomaly_count": len(analysis_results.get('mathematical_validation', {}).get('statistical_flags', [])),
                "visual_deviation_count": analysis_results.get('spatial_fingerprinting', {}).get('total_deviations', 0)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(training_record, f, indent=2)
        
        return {"success": True, "filepath": str(filepath), "filename": filename}
        
    except Exception as e:
        return {"success": False, "error": str(e), "filepath": None}
