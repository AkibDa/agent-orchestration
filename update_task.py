with open('/Users/skakibahammed/.gemini/antigravity-ide/brain/1fdd20a3-242e-4688-acb9-94ef592ca21e/task.md', 'r') as f:
    text = f.read()

text = text.replace('`[/]` 2. **INCOIS', '`[x]` 2. **INCOIS')
text = text.replace('`[ ]` 3. **INCOIS', '`[x]` 3. **INCOIS')
text = text.replace('`[ ]` 4. **PFZ', '`[x]` 4. **PFZ')
text = text.replace('`[ ]` 5. **Validation', '`[x]` 5. **Validation')

with open('/Users/skakibahammed/.gemini/antigravity-ide/brain/1fdd20a3-242e-4688-acb9-94ef592ca21e/task.md', 'w') as f:
    f.write(text)
