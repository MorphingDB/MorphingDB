from operator import imod
import os
import subprocess


# 定义要执行的脚本列表
scripts_to_run = {
    #'import_dataset.py': {}
    'morphingdb_test.py': {}
    #'evadb_test.py': {}
}


dir_list = ['morphingdb_test/series_test']
#'image_test', 'text_test', 

# 设置项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

def run_scripts_in_directory(directory, scripts):
    for dir in dir_list:
        for root, dirs, files in os.walk(os.path.join(directory, dir)):
            for script_name in scripts:
                script_path = os.path.join(root, script_name)
                if script_name in files and os.path.isfile(script_path):
                    print(f"Running {script_path}...")
                    os.chdir(root)
                    print(os.getcwd())
                    try:
                        # 执行脚本并检查返回状态，出错时捕获异常
                        subprocess.run(['python3', script_path], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running {script_path}: {e}")
                    except Exception as e:
                        print(f"Unexpected error running {script_path}: {e}")
                # else:
                #     print(f"Script {script_name} not found in {root}, skipping...")

if __name__ == "__main__":
    #run_scripts_in_directory(project_root, scripts_to_run.keys())
    # series test
    from morphingdb_test.series_test.slice_test.morphingdb_test import slice_all_test 
    slice_all_test()
    from morphingdb_test.series_test.swarm_test.morphingdb_test import swarm_all_test
    swarm_all_test()
    from morphingdb_test.series_test.year_predict_test.morphingdb_test import year_predict_all_test
    year_predict_all_test()
    
    # image test
    from morphingdb_test.image_test.cifar10.morphingdb_test import cifar10_all_test
    cifar10_all_test()
    from morphingdb_test.image_test.imagenet.morphingdb_test import imagenet_all_test
    imagenet_all_test()
    from morphingdb_test.image_test.stanford_dogs.morphingdb_test import stanford_dogs_all_test
    stanford_dogs_all_test()

    # text test
    from morphingdb_test.text_test.financial_phrasebank.morphingdb_test import financial_phrasebank_all_test
    financial_phrasebank_all_test()
    from morphingdb_test.text_test.imdb.morphingdb_test import imdb_all_test
    imdb_all_test()
    from morphingdb_test.text_test.sst2.morphingdb_test import sst2_all_test
    sst2_all_test()

    # muti test
    from morphingdb_test.muti_query.morphingdb_test import muti_query_all_test
    muti_query_all_test()

    # auto ml 还有点问题
    from morphingdb_test.auto_ml.morphingdb_test import auto_ml_test
    auto_ml_test()
    
    # cost model 
    from morphingdb_test.cost_model_test.cost_model_test import cost_model_test
    cost_model_test()

    # batch test
    from morphingdb_test.batch_test.morphingdb_test import batch_all_test
    batch_all_test()

    # api server
    from morphingdb_test.api_server.api_test import api_load_model_test
    api_load_model_test()



    # evadb 
    from morphingdb_test.series_test.slice_test.evadb_test import evadb_slioe_test
    evadb_slioe_test()
    from morphingdb_test.series_test.year_predict_test.evadb_test import evadb_year_predict_test
    evadb_year_predict_test()

    from morphingdb_test.image_test.cifar10.evadb_test import evadb_cifar_test
    evadb_cifar_test()
    from morphingdb_test.image_test.imagenet.evadb_test import evadb_imagenet_test
    evadb_imagenet_test()
    from morphingdb_test.image_test.stanford_dogs.evadb_test import evadb_stanford_dogs_test
    evadb_stanford_dogs_test()

    from morphingdb_test.text_test.financial_phrasebank.evadb_test import evadb_financial_phrasebank_test
    evadb_financial_phrasebank_test()
    from morphingdb_test.text_test.imdb.evadb_test import evadb_imdb_test
    evadb_imdb_test()
    from morphingdb_test.text_test.sst2.evadb_test import evadb_sst2_test
    evadb_sst2_test()

    from morphingdb_test.muti_query.evadb_test import evadb_muti_query_test
    evadb_muti_query_test()
