"""
Create minimal stubs for packages that pathway.xpacks.llm.parsers imports
unconditionally at module level, but that this service never actually uses.
This avoids installing huge optional packages (e.g. unstructured = 50+ MB).
"""
import pathlib, textwrap

base = pathlib.Path('/usr/local/lib/python3.10/site-packages')

packages = [
    # unstructured
    'unstructured',
    'unstructured/file_utils',
    'unstructured/documents',
    'unstructured/partition',
    'unstructured/chunking',
    # docling_core
    'docling_core',
    'docling_core/types',
    'docling_core/types/doc',
    'docling_core/transforms',
    'docling_core/transforms/chunker',
    'docling_core/types/io',
    # docling
    'docling',
    'docling/chunking',
    'docling/datamodel',
    'docling/datamodel/base_models',
    'docling/document_converter',
]

for pkg in packages:
    p = base / pkg
    p.mkdir(parents=True, exist_ok=True)
    init = p / '__init__.py'
    if not init.exists():
        init.touch()

# unstructured stubs
(base / 'unstructured/file_utils/filetype.py').write_text(textwrap.dedent("""
class FileType:
    pass
def detect_filetype(*a, **kw):
    return None
"""))
(base / 'unstructured/documents/elements.py').write_text("class Element: pass\n")
(base / 'unstructured/partition/auto.py').write_text("def partition(*a, **kw): return []\n")
(base / 'unstructured/partition/common.py').write_text("")
(base / 'unstructured/chunking/basic.py').write_text("def chunk_elements(*a, **kw): return []\n")
(base / 'unstructured/chunking/title.py').write_text("def chunk_by_title(*a, **kw): return []\n")

# docling/chunking — imported by pathway.xpacks.llm._parser_utils
(base / 'docling/chunking/__init__.py').write_text(textwrap.dedent("""
class HierarchicalChunker:
    def __init__(self, *a, **kw): pass
    def chunk(self, *a, **kw): return iter([])

class HybridChunker:
    def __init__(self, *a, **kw): pass
    def chunk(self, *a, **kw): return iter([])
"""))

# docling/document_converter — also used by pathway parsers
(base / 'docling/document_converter/__init__.py').write_text(textwrap.dedent("""
class PdfFormatOption:
    def __init__(self, *a, **kw): pass

class DocumentConverter:
    def __init__(self, *a, **kw): pass
    def convert(self, *a, **kw): return None
"""))

# docling/datamodel stubs
(base / 'docling/datamodel/__init__.py').write_text("")
(base / 'docling/datamodel/pipeline_options.py').write_text(textwrap.dedent("""
class PdfPipelineOptions:
    def __init__(self, *a, **kw): pass
"""))
(base / 'docling/datamodel/base_models/__init__.py').write_text(textwrap.dedent("""
class InputFormat:
    pass
class DocumentStream:
    def __init__(self, *a, **kw): pass
"""))

# docling_core/transforms/chunker — needs BaseChunk, DocChunk, DocMeta
(base / 'docling_core/transforms/chunker/__init__.py').write_text(textwrap.dedent("""
class BaseChunk:
    def __init__(self, *a, **kw): pass

class DocChunk(BaseChunk):
    def __init__(self, *a, **kw): pass

class DocMeta:
    def __init__(self, *a, **kw): pass
"""))
(base / 'docling_core/transforms/chunker/hierarchical_chunker.py').write_text("class HierarchicalChunker: pass\n")

# docling_core/types/doc/document — needs all the item classes
(base / 'docling_core/types/doc/__init__.py').write_text(textwrap.dedent("""
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
"""))
(base / 'docling_core/types/doc/document.py').write_text(textwrap.dedent("""
class DoclingDocument:
    def __init__(self, *a, **kw): pass

class DocItem:
    def __init__(self, *a, **kw): pass

class CodeItem(DocItem): pass
class ListItem(DocItem): pass
class PictureItem(DocItem): pass
class SectionHeaderItem(DocItem): pass
class TableItem(DocItem): pass
class TextItem(DocItem): pass
class TitleItem(DocItem): pass

LevelNumber = int
"""))

# docling_core/types/doc/labels — needs DocItemLabel
(base / 'docling_core/types/doc/labels.py').write_text(textwrap.dedent("""
class DocItemLabel:
    pass
"""))

(base / 'docling_core/types/io/__init__.py').write_text("")

print("All stubs created successfully.")
