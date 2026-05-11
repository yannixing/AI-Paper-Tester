pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        MODELSCOPE_API_KEY = credentials('ms-883ddad7-ee94-4cac-b1d3-aafbb3761b20')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    #set -e
                    python3 -m venv agent
                    . agent/bin/activate
                    python -m pip install --upgrade pip

                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    else
                        # 兜底安装（至少保证 pytest 和你当前 agent 依赖可用）
                        pip install pytest langchain langchain-openai langchain-chroma langchain-huggingface sentence-transformers chromadb
                    fi
                 '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                set -e
                . agent/bin/activate
                pytest test_agent.py -q --junitxml=test-results.xml
                 '''
            }
        }

        stage('Result') {
            steps {
                echo '测试完成'
            }
        }
    }

    post {
        always {
            junit testResults: 'test-results.xml', allowEmptyResults: true
            archiveArtifacts artifacts: 'test-results.xml', allowEmptyArchive: true
            echo '无论成功或失败，均已收集测试报告'
        }
    }
}
