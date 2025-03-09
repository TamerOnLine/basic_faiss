import os

def get_scripts():
    """جلب جميع سكريبتات بايثون داخل المجلد الحالي"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        str(index): os.path.join(scripts_dir, file)
        for index, file in enumerate(os.listdir(scripts_dir))
        if file.endswith(".py") and file != os.path.basename(__file__)  # استثناء runner.py نفسه
    }

def run_script():
    """عرض السكريبتات المتاحة وتنفيذ السكريبت المختار"""
    scripts = get_scripts()

    if not scripts:
        print("\n❌ لا توجد سكريبتات Python للتشغيل في هذا المجلد.")
        return

    print("\n📜 Select the script you want to run:\n")
    for key, script in scripts.items():
        print(f"{key}. {script}")

    choice = input("\n🔹 Enter the number of the script to execute: ").strip()

    if choice in scripts:
        script_to_run = scripts[choice]
        print(f"\n🚀 Running {script_to_run}...\n")
        os.system(f"python {script_to_run}")
    else:
        print("\n❌ Invalid choice. Please select a valid number.")

if __name__ == "__main__":
    run_script()
