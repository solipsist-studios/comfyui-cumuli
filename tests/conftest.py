# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Keep pytest from walking up into the pack root, which carries an __init__.py
because ComfyUI imports the whole directory as a package."""
