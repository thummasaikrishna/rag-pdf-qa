"""
Generates two small sample PDFs used for testing/demoing the RAG app.
Run: python scripts/make_sample_pdfs.py
"""
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_pdfs")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def make_pdf(path, title, pages_content):
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                             topMargin=1 * inch, bottomMargin=1 * inch)
    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))
    for i, content in enumerate(pages_content):
        for para in content:
            story.append(Paragraph(para, styles["BodyText"]))
            story.append(Spacer(1, 0.15 * inch))
        if i < len(pages_content) - 1:
            story.append(PageBreak())
    doc.build(story)
    print(f"Wrote {path}")


company_policy = [
    [
        "Company Leave Policy",
        "All full-time employees are entitled to 18 days of paid annual leave per calendar year, "
        "accrued monthly at a rate of 1.5 days per month.",
        "Sick leave is granted separately, up to 10 days per year, and does not carry over to the next year.",
        "Employees must submit leave requests through the HR portal at least 3 working days in advance, "
        "except in cases of medical emergency.",
    ],
    [
        "Remote Work Guidelines",
        "Employees may work remotely up to 2 days per week with manager approval.",
        "Fully remote arrangements require VP-level sign-off and are reviewed every 6 months.",
        "All remote employees must be reachable during core hours, defined as 10:00 AM to 4:00 PM local time.",
    ],
    [
        "Expense Reimbursement",
        "Business travel expenses must be submitted within 30 days of the trip using the Expensify tool.",
        "Meals are reimbursed up to $60 per day domestically and $90 per day internationally.",
        "Reimbursements are processed within 10 business days of approval.",
    ],
]

product_manual = [
    [
        "SmartBrew Coffee Maker - User Manual",
        "The SmartBrew 3000 supports three brewing modes: Standard, Bold, and Cold Brew.",
        "To descale the machine, run the descaling cycle every 60 brew cycles using the descale button "
        "on the top panel.",
    ],
    [
        "Troubleshooting",
        "If the machine displays Error 4, this indicates the water reservoir is not seated correctly. "
        "Remove and reinsert the reservoir firmly until it clicks.",
        "If brewing takes longer than 5 minutes, the machine may need descaling.",
    ],
    [
        "Warranty Information",
        "The SmartBrew 3000 comes with a 2-year limited warranty covering manufacturing defects.",
        "The warranty does not cover damage caused by using non-filtered water or third-party descaling agents.",
    ],
]

make_pdf(os.path.join(OUT_DIR, "company_policy.pdf"), "Company Policy Handbook",
          company_policy)
make_pdf(os.path.join(OUT_DIR, "product_manual.pdf"), "SmartBrew 3000 Manual",
          product_manual)
