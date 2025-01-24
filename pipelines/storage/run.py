from tools.aws.connector import S3Connector

S3Connector().upload(
    "/home/razvantalexandru/Documents/Projects/NeuralBits/vision-ai-course/pipelines/storage/xx.txt", "RawData/test.txt"
)
