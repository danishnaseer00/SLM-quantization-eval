"""Deprecated compatibility entry point for the human-annotation audit.

No automated adjudication is performed.  The paper reports the two original
human annotation sets separately and preserves their disagreements.  Use
``analysis/analyze_human_annotations.py`` directly in new workflows.
"""

from analyze_human_annotations import main


if __name__ == "__main__":
    main()