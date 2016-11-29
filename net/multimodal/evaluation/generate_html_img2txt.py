"""

words_pred.json is generated with i2t_predict
"""


import json

split = 'test'
voc_mode = 'zappos'
vis = 'fc7'
model = 'bilda'
n_topic = 200
alpha = '0.12'

ckpoint_path = '../data/fashion53k/result_plots/'

fname = ckpoint_path + 'words_pred_katrien.json'


template = """<div class="row rnndemo">
<div class="col-sm-3">
  <img src="{}" class="demoimg">
  <br>
  "{}"
  </div>
</div>
"""

out_html = ''

excluded = {'dress', 'evening', 'dresses', 'back', 'please', 'bones', 'made',
            'subject', 'lined', 'available', 'cause', 'full',
            'objects', 'obey', 'made', 'place', 'fully', 'welcome',
            'strict', 'special', 'thanks', 'women', 'woman', 'zipper', 'prom'}

data = json.load(open(fname, 'r'))
for item in data['items']:
    # print item['folder']
    # print item['img_filename']
    # print item['words_predicted']

    img_dir = '/Users/susanaparis/Documents/Belgium/IMAGES_plus_TEXT/DATASETS/' + item['folder'] + item['img_filename']
    predicted_text = item['words_predicted']

    temp_text = predicted_text.split(' ')
    new_predicted_text = ' '.join([w for w in temp_text if w not in excluded])

    temp_html = template.format(img_dir, new_predicted_text)

    out_html += temp_html

print out_html




