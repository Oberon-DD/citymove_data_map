"""Redaction rules applied to free text before publication.

Catalogue metadata harvested from the city portals sometimes names the staff
member who maintains a dataset ("Management: <name> Contact: <address>",
"Responsible: <name>"). These fields add nothing to the inventory, so the
public copies replace them. Everything else is left as harvested.

The labels are matched case-sensitively so that running text such as
"area with effective nature management: ..." is not touched.
"""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]*)*")
# Value of a maintainer field: runs up to the next "Contact:" label, a line break or the end.
MAINTAINER = re.compile(r"\b(Management|Managed by|Maintained by)(\s*:\s*)(.+?)(?=\s+Contact\s*:|[\r\n]|$)")
RESPONSIBLE = re.compile(r"\b(Responsible|Responsable)(\s*:\s*)[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}")
RESPONSIBLE_NO_COLON = re.compile(r"\b(Responsible|Responsable)(\s+)[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?=\s*$|\s*[.;,\-])")
NAME_BEFORE_EMAIL = re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+(?=[\w.+-]+@[\w-]+)")
# A contact value that survived the e-mail rule (address cut off by the catalogue's length limit).
CONTACT_LEFTOVER = re.compile(r"\b(Contact)(\s*:\s*)(?!\[email redacted\])\S+")

PERSON = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:van|de|der|den|Van|De))?(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,2}"
PERSON_IN_FIELD = [
    re.compile(r"\b(?:Management|Managed by|Maintained by)\s*:\s*(" + PERSON + r")(?=\s+Contact\s*:|[\r\n]|$)"),
    re.compile(r"\b(?:Responsible|Responsable)\s*:\s*(" + PERSON + r")\b"),
    re.compile(r"\b(?:Responsible|Responsable)\s+(" + PERSON + r")(?=\s*$|\s*[.;,\-])"),
]

EMAIL_MARK = "[email redacted]"
NAME_MARK = "[redacted]"


def collect_names(texts):
    """Personal names that appear in maintainer fields anywhere in `texts`.

    Synthetic paraphrases of the same metadata ("managed by <name> since 2015")
    carry no field label, so the names found here are redacted wherever they
    occur. The list is built at run time from the private source and never saved.
    """
    names = set()
    for t in texts:
        if isinstance(t, str):
            for rx in PERSON_IN_FIELD:
                names.update(m.group(1).strip() for m in rx.finditer(t))
    return names


def redact(text, names=()):
    """Return (clean_text, n_substitutions). `names` come from collect_names()."""
    if not isinstance(text, str) or not text:
        return text, 0
    n = 0
    for name in sorted(names, key=len, reverse=True):
        text, k = re.subn(r"\b" + re.escape(name) + r"\b", NAME_MARK, text)
        n += k
    text, k = MAINTAINER.subn(lambda m: f"{m.group(1)}{m.group(2)}{NAME_MARK}", text)
    n += k
    text, k = RESPONSIBLE.subn(lambda m: f"{m.group(1)}{m.group(2)}{NAME_MARK}", text)
    n += k
    text, k = RESPONSIBLE_NO_COLON.subn(lambda m: f"{m.group(1)}{m.group(2)}{NAME_MARK}", text)
    n += k
    text, k = NAME_BEFORE_EMAIL.subn(f"{NAME_MARK} ", text)
    n += k
    text, k = EMAIL.subn(EMAIL_MARK, text)
    n += k
    text, k = CONTACT_LEFTOVER.subn(lambda m: f"{m.group(1)}{m.group(2)}{NAME_MARK}", text)
    n += k
    return text, n
