import preprocess
import back_ml
class Predictor():
    def __init__(self):
        self.model = back_ml.model('save.pkl','embs_df_train_umap.pkl')
        self.model.test()
        pass


    def get_preds(self,images):
        images_=[]
        img = preprocess.pil_from_base(images)
        for i in range(len(images)):
            image = img[i]
            pos = preprocess.get_geo(image)
            images_.append({'img':image,'pos':pos})
        preds = self.model.get_preds(images_)
        print(preds)
