import os

def get_runner_scripts_in_subdirectories():
    """البحث عن جميع سكريبتات runner.py في المجلدات الفرعية"""
    project_dir = os.path.dirname(os.path.abspath(__file__))  # المجلد الرئيسي للمشروع
    runner_scripts = {}

    # السير في المجلدات الفرعية
    for root, dirs, files in os.walk(project_dir):
        if 'runner.py' in files:  # التحقق من وجود ملف runner.py
            runner_scripts[root] = os.path.join(root, 'runner.py')

    return runner_scripts

def run_script():
    """عرض السكريبتات المتاحة وتنفيذ السكريبت المختار"""
    runner_scripts = get_runner_scripts_in_subdirectories()

    if not runner_scripts:
        print("\n❌ لا توجد سكريبتات runner.py للتشغيل في المجلدات الفرعية.")
        return

    print("\n📜 Select the script you want to run:\n")
    for index, (key, script) in enumerate(runner_scripts.items()):
        print(f"{index + 1}. {script}")

    try:
        choice = int(input("\n🔹 Enter the number of the script to execute: ").strip()) - 1
        if 0 <= choice < len(runner_scripts):
            script_to_run = list(runner_scripts.values())[choice]
            print(f"\n🚀 Running {script_to_run}...\n")
            os.system(f"python {script_to_run}")
        else:
            print("\n❌ Invalid choice. Please select a valid number.")
    except ValueError:
        print("\n❌ Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    run_script()
