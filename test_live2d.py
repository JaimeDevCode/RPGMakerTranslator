"""Check if Live2D plugin commands in map event data got translated."""
import pathlib
import json

game = pathlib.Path(r"F:\Users\meganano202\Desktop\test\Japanese\RJ01519859")
data_dir = game / "www" / "data"
backup_dir = game / "translation_project" / "backup"

def get_plugin_cmds_from_list(cmd_list):
    """Extract code-356 plugin commands containing 'live2d'."""
    cmds = []
    for cmd in (cmd_list or []):
        if cmd.get("code") == 356:
            for p in cmd.get("parameters", []):
                if isinstance(p, str) and "live2d" in p.lower():
                    cmds.append(p)
    return cmds

# Check CommonEvents.json
ce = json.loads((data_dir / "CommonEvents.json").read_text(encoding="utf-8"))
ce_bak = json.loads((backup_dir / "CommonEvents.json").read_text(encoding="utf-8"))

print("=== CommonEvents Live2D plugin commands ===")
for i in range(min(len(ce), len(ce_bak))):
    if ce[i] is None or ce_bak[i] is None:
        continue
    cur_cmds = get_plugin_cmds_from_list(ce[i].get("list"))
    bak_cmds = get_plugin_cmds_from_list(ce_bak[i].get("list"))
    if cur_cmds != bak_cmds:
        print(f"  CE {i}: CHANGED")
        for j, (c, b) in enumerate(zip(cur_cmds, bak_cmds)):
            if c != b:
                print(f"    [{j}] ORIG: {b[:80]}")
                print(f"    [{j}] NOW:  {c[:80]}")

# Check Map002.json
for map_name in ["Map002.json", "Map004.json"]:
    mp = json.loads((data_dir / map_name).read_text(encoding="utf-8"))
    mp_bak_f = backup_dir / map_name
    if not mp_bak_f.exists():
        continue
    mp_bak = json.loads(mp_bak_f.read_text(encoding="utf-8"))

    print(f"\n=== {map_name} Live2D plugin commands ===")
    for ev in (mp.get("events") or []):
        if ev is None:
            continue
        eid = ev.get("id")
        bak_ev = None
        for bev in (mp_bak.get("events") or []):
            if bev and bev.get("id") == eid:
                bak_ev = bev
                break
        if not bak_ev:
            continue
        for pi, (page, bpage) in enumerate(zip(ev.get("pages", []), bak_ev.get("pages", []))):
            cur_cmds = get_plugin_cmds_from_list(page.get("list"))
            bak_cmds = get_plugin_cmds_from_list(bpage.get("list"))
            if cur_cmds != bak_cmds:
                print(f"  Event {eid} page {pi}: CHANGED")
                for j, (c, b) in enumerate(zip(cur_cmds, bak_cmds)):
                    if c != b:
                        print(f"    ORIG: {b[:80]}")
                        print(f"    NOW:  {c[:80]}")
                break  # just show first diff per event
