"""
run to anonymize
"""
import os
import shutil


RAW_DATA_PATH = '../mount'

NEG_DIRS = ['Non referral1', 'Non Referral2', 'NonReferral3']
POS_DIRS = ['Referral1', 'Referral2', 'Referral3']

def anonymize():
    for label, top_dirs in [('pos', POS_DIRS), ('neg', NEG_DIRS)]:
        for top_dir in top_dirs:
            top_dir_path = os.path.join(RAW_DATA_PATH, top_dir)
            print('=================')
            print(top_dir_path)
            for case_dir in os.listdir(top_dir_path):
                case_dir_path = os.path.join(top_dir_path, case_dir)
                print(case_dir)

                for visit_dir in os.listdir(case_dir_path):
                    if visit_dir.startswith('Case'):
                        print(f'Skipping directory: {case_dir_path}/{visit_dir}')
                        continue
                    
                    visit_dir_path = os.path.join(case_dir_path, visit_dir)
                    patient_id, date = visit_dir.split()
                    print(patient_id)

                    for file in os.listdir(visit_dir_path):
                        patient_id_2, *rest = file.split('_')
                        assert patient_id == patient_id_2 or patient_id_2.startswith('Case')
                        new_file = '_'.join([case_dir] + rest)

                        old_path = os.path.join(visit_dir_path, file)
                        new_path = os.path.join(visit_dir_path, new_file)
                        print(f'Renaming: {old_path} -> {new_path}')
                        try:
                            os.rename(old_path, new_path)
                        except Exception as e:
                            print('!!!!!! ERROR WHEN RENAMING FILE !!!!!!')
                            print(e)

                    # Google Cloud Storage does not support 'mv'
                    new_dir_path = os.path.join(case_dir_path, f'{case_dir} {date}')
                    print(f'Copying dir: {visit_dir_path} -> {new_dir_path}')
                    try:
                        shutil.copytree(visit_dir_path, new_dir_path)
                    except Exception as e:
                        print('!!!!!! ERROR WHEN COPYING DIR !!!!!!')
                        print(e)

                    print(f'Deleting dir: {visit_dir_path}')
                    try:
                        shutil.rmtree(visit_dir_path)
                    except Exception as e:
                        print('!!!!!! ERROR WHEN DELETING DIR !!!!!!')
                        print(e)
                        
if __name__ == '__main__' and input('Are you sure? ') == 'y':
    anonymize()
