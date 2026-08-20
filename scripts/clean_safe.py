import re

with open('app/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Sidebar toggle
s1 = '''    # Edge AI mode toggle
    edge_ai_mode = st.toggle("⚡ Edge AI Mode", value=False,
                             help="Switch to lightweight StudentDNNet for fast mobile-class inference (~8ms CPU)")
    if edge_ai_mode:
        st.sidebar.caption("🟢 Using Student (512-d)")
    else:
        st.sidebar.caption("🔵 Using Teacher (1024-d)")'''
content = content.replace(s1, '    edge_ai_mode = False')

# 2. Sidebar warning
s2 = '''    if edge_ai_mode:
        st.info("⚡ **Edge AI Mode Active** — Using StudentDNNet (MobileNetV3-Small, 512-d). "
                "Searching student-model embeddings only. "
                "If no student embeddings exist, register dogs with Edge AI mode enabled first.")
        if not os.path.exists("checkpoints/student_best.pth"):
            st.warning("⚠️ No distilled student weights found. Using ImageNet-pretrained backbone only — "
                       "accuracy may be lower than the teacher model.")'''
content = content.replace(s2, '')

# 3. Sidebar menu
content = content.replace('"🗄️ Database",\n        "⚡ Edge AI",', '"🗄️ Database",')
content = content.replace('"🗄️ Database",\r\n        "⚡ Edge AI",', '"🗄️ Database",')

# 4. Database tabs
content = re.sub(r'tab_schema, tab_sql, tab_analytics = st\.tabs\(\[.*?\]\)', '', content, flags=re.DOTALL)
content = re.sub(r'with tab_schema:', 'if False:', content)
content = re.sub(r'with tab_sql:', 'if False:', content)
content = re.sub(r'with tab_analytics:', 'if True:', content)

# 5. Remove Edge AI block completely and append JSON export
parts = content.split('# ==============================================================================\n# PAGE: EDGE AI')
if len(parts) == 1:
    parts = content.split('# ==============================================================================\r\n# PAGE: EDGE AI')

new_content = parts[0]

export_json_block = """
    st.divider()
    st.subheader("📱 Export Gallery for Mobile")
    st.markdown("Export the embeddings to a JSON file that can be imported directly into the DogID Edge Android app.")
    
    if st.button("📄 Generate Gallery JSON", type="secondary"):
        conn = get_conn()
        rows = conn.execute(\"\"\"
            SELECT d.name, e.embedding
            FROM dogs d
            JOIN embeddings e ON e.dog_id = d.id
            WHERE e.model_type = 'student'
        \"\"\").fetchall()
        conn.close()
        
        if not rows:
            st.warning("⚠️ No student embeddings found in the database. Run batch_reenroll_student.py first.")
        else:
            export_data = []
            for name, emb_blob in rows:
                import struct
                num_floats = len(emb_blob) // 4
                floats = struct.unpack(f"<{num_floats}f", emb_blob)
                export_data.append({
                    "name": name,
                    "embedding": list(floats)
                })
            
            import json
            json_str = json.dumps(export_data)
            st.success(f"✅ Generated JSON with {len(export_data)} embeddings.")
            st.download_button(
                "⬇️ Download Gallery JSON",
                json_str,
                file_name="dogid_gallery.json",
                mime="application/json",
            )
"""

new_content += export_json_block

with open('app/app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Safely cleaned app.py")
