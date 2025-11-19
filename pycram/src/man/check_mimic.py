import xml.etree.ElementTree as ET
import os

def check_urdf_limits(urdf_file: str):
    if not os.path.isfile(urdf_file):
        print(f" File not found: {urdf_file}")
        return

    tree = ET.parse(urdf_file)
    root = tree.getroot()

    joints = {}
    mimic_relations = []

    print(f"\n Checking URDF: {urdf_file}\n")

    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        limit = joint.find("limit")

        # Store joint info for later mimic check
        joints[name] = {"type": jtype, "limit": limit}

        # Check limits depending on type
        if jtype in ["revolute", "prismatic"]:
            if limit is None:
                print(f"[MISSING] {jtype} joint '{name}' has no <limit> tag")
            else:
                missing = []
                if limit.get("lower") is None: missing.append("lower")
                if limit.get("upper") is None: missing.append("upper")
                if limit.get("velocity") is None: missing.append("velocity")
                if limit.get("effort") is None: missing.append("effort")

                if missing:
                    print(f"[INCOMPLETE] {jtype} joint '{name}' missing: {', '.join(missing)}")

        elif jtype == "continuous":
            if limit is None:
                print(f"[MISSING] continuous joint '{name}' has no <limit> tag (needs velocity + effort)")
            else:
                if limit.get("velocity") is None or limit.get("effort") is None:
                    print(f"[INCOMPLETE] continuous joint '{name}' missing velocity/effort")

        elif jtype in ["fixed", "floating", "planar"]:
            if limit is not None:
                print(f"[WARNING] {jtype} joint '{name}' should not have a <limit> tag")

        # Handle mimic joints
        mimic = joint.find("mimic")
        if mimic is not None:
            ref_joint = mimic.get("joint")
            mimic_relations.append((name, ref_joint))

    # Now check mimic consistency
    if mimic_relations:
        print("\n Checking mimic joints...\n")
        for mimic_name, ref_joint in mimic_relations:
            if ref_joint not in joints:
                print(f"[ERROR] Mimic joint '{mimic_name}' references unknown joint '{ref_joint}'")
            else:
                ref_info = joints[ref_joint]
                if ref_info["limit"] is None and ref_info["type"] not in ["fixed", "floating", "planar"]:
                    print(f"[ERROR] Mimic joint '{mimic_name}' references '{ref_joint}' which has NO <limit> tag")

    print("\n Check finished.\n")

if __name__ == "__main__":

    check_urdf_limits('/home/zakaria/workspace/ros/src/pycram/resources/robots/tracy.urdf')
