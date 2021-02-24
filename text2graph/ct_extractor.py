#!/usr/bin/python3

# =============================================================================
# CrowdTangle Utils
# =============================================================================
#
# Miscellaneous utility functions to be used with CrowdTangle Link.
# @Author: Brayan Rodriguez <bradrd2009jp@gmail.com>
# @Organization: LIIT-UNED 2020

#TODO:
#Include module for make search with CrowdTangle API.

import urllib.request, json
import pandas as pd
import tldextract

#Constantes:
main_url = 'https://api.crowdtangle.com'

__all__ = ['get_dict', 'get_json', 'ctdatapost_', 'ctdatalink_', 'get_ct_data', 'ct_lists', 'ct_accounts', 'ct_leaderboard_data', 'ct_posts', 'ct_search_data']

def get_dict(json_data):
    return {key: json_data[key] for key in  json_data.keys()}

def get_json(url_data):
    with urllib.request.urlopen(url_data) as url:
        data = json.loads(url.read().decode())
    return data

class ctdatapost_():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def notes(self):
        try:
            return self.dict_data['notes']
        except KeyError:
            print("There was not included \'notes\' in this searching return")
            return ''
    def post(self):
        result_dict = get_dict(self.result())
        return result_dict['posts']
    def platform_id(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['platformId']
    def date(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['date']
    def message(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['message']
        except KeyError:
            print("There was not included \'message\' in this searching return")
            return ''
    def title(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['title']
        except KeyError:
            print("There was not included \'title\' in this searching return")
            return ''
    def ct_id(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['id']
    def link(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['link']
    def post_url(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['postUrl']
    def domain(self):
        ext = tldextract.extract(self.link())
        return ext.domain
    def type(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['type']
    def media(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['media']
    def media_type(self):
        media_dict = get_dict(self.media()[0])
        return media_dict.get('type')
    def media_url(self):
        media_dict = get_dict(self.media()[0])
        return media_dict.get('full')   
    def statistics(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['statistics']
    def statistics_df(self):
        stat_dict = get_dict(self.statistics())
        columns = ['platformId']
        value_lst = [self.platform_id()]
        for key, value in stat_dict['actual'].items():
            columns.append('actual_%s'%key)
            value_lst.append(value)
        for key, value in stat_dict['expected'].items():
            columns.append('expected_%s'%key)
            value_lst.append(value)
        df = pd.DataFrame([value_lst], columns=columns)
        return df
    def history(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['history']
        except KeyError:
            print("There was not included \'history\' in this searching return")
            return 0
    def history_df(self):
        try:
            post_dict = get_dict(self.post()[0])
            df_prev = pd.DataFrame(post_dict['history'])
            df_final = pd.DataFrame()
            lst_aux = ['likeCount', 'shareCount', 'commentCount', 'loveCount', 'wowCount', 'hahaCount', 'sadCount', 'angryCount', 'thankfulCount', 'careCount']
            for i in lst_aux:
                df_final['actual_%s'%i] = [k.get(i) for k in df_prev['actual']]
            for i in lst_aux:
                df_final['expected_%s'%i] = [k.get(i) for k in df_prev['expected']]
            df_final['timestep'] = df_prev['timestep'].tolist()
            df_final['date'] = df_prev['date'].tolist()
            df_final['score'] = df_prev['score'].tolist()
            return df_final
        except KeyError:
            print("There was not included \'history\' in this searching return")
            return 0

class ctdatalink_():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def notes(self):
        try:
            return self.dict_data['notes']
        except KeyError:
            print("There was not included \'notes\' in this searching return")
            return ''
    def post(self):
        result_dict = get_dict(self.result())
        return result_dict['posts']
    def platform_id(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['platformId']
    def date(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['date']
    def message(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['message']
        except KeyError:
            print("There was not included \'message\' in this searching return")
            return ''
    def title(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['title']
        except KeyError:
            print("There was not included \'title\' in this searching return")
            return ''
    def caption(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['caption']
    def link(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['link']
    def post_url(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['postUrl']
    def domain(self):
        ext = tldextract.extract(self.link())
        return ext.domain
    def post_url(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['postUrl']
    def media(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['media']
    def media_type(self):
        media_dict = get_dict(self.media()[0])
        return media_dict.get('type')
    def media_url(self):
        media_dict = get_dict(self.media()[0])
        return media_dict.get('full')   
    def statistics(self):
        post_dict = get_dict(self.post()[0])
        return post_dict['statistics']
    def statistics_df(self):
        stat_dict = get_dict(self.statistics())
        columns = ['platformId']
        value_lst = [self.platform_id()]
        for key, value in stat_dict['actual'].items():
            columns.append('actual_%s'%key)
            value_lst.append(value)
        for key, value in stat_dict['expected'].items():
            columns.append('expected_%s'%key)
            value_lst.append(value)
        df = pd.DataFrame([value_lst], columns=columns)
        return df
    def history(self):
        try:
            post_dict = get_dict(self.post()[0])
            return post_dict['history']
        except KeyError:
            print("There was not included \'history\' in this searching return")
            return 0
    def history_df(self):
        try:
            post_dict = get_dict(self.post()[0])
            df_prev = pd.DataFrame(post_dict['history'])
            df_final = pd.DataFrame()
            lst_aux = ['likeCount', 'shareCount', 'commentCount', 'loveCount', 'wowCount', 'hahaCount', 'sadCount', 'angryCount', 'thankfulCount', 'careCount']
            for i in lst_aux:
                df_final['actual_%s'%i] = [k.get(i) for k in df_prev['actual']]
            for i in lst_aux:
                df_final['expected_%s'%i] = [k.get(i) for k in df_prev['expected']]
            df_final['timestep'] = df_prev['timestep'].tolist()
            df_final['date'] = df_prev['date'].tolist()
            df_final['score'] = df_prev['score'].tolist()
            return df_final
        except KeyError:
            print("There was not included \'history\' in this searching return")
            return 0

class ct_lists():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def list_of_dict(self):
        result_dict = get_dict(self.result())
        return result_dict['lists']
    def list_df(self):
        df_final = pd.DataFrame()
        lst_aux = ['id', 'title', 'type']
        for i in lst_aux:
            df_final[i] = [k.get(i) for k in self.list_of_dict()]       
        return df_final
    def lists_of_id(self):
        return self.list_df()['id'].tolist()

class ct_accounts():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def list_of_accounts(self):
        result_dict = get_dict(self.result())
        return result_dict['accounts']
    def accounts_df(self):
        df_final = pd.DataFrame()
        lst_aux = ['id', 'name', 'handle', 'profileImage', 'suscriberCount', 'url', 'platformId', 'accountType', 'pageAdminTopCountry', 'verified']
        for i in lst_aux:
            df_final[i] = [k.get(i) for k in self.list_of_accounts()]       
        return df_final

class ct_leaderboard_data():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def list_of_accounts(self):
        post_dict = get_dict(self.result())
        return post_dict['accountStatistics']
    def return_list(self, key, dict_of_dicts):
        return [k.get(key) for k in dict_of_dicts]   
    def get_df(self):
        df_prev = pd.DataFrame()
        df_final = pd.DataFrame()
        lst_aux = ['account', 'summary', 'subscriberData', ]
        for i in lst_aux:
            df_prev[i] = [k.get(i) for k in self.list_of_accounts()]
        lst_acc = ['id', 'name', 'handle', 'subscriberCount', 'url', 'platformId', 'pageAdminTopCountry', 'verified']
        for i in lst_acc:
            df_final[i] = self.return_list(i, df_prev['account'])
        lst_sum = ['likeCount', 'loveCount', 'hahaCount', 'wowCount', 'thankfulCount',  'angryCount', 'sadCount', 'shareCount', 'commentCount', 'totalInteractionCount', 'interactionRate']
        for i in lst_sum:
            df_final[i] = self.return_list(i, df_prev['summary'])
        lst_sbd = ['initialCount', 'finalCount']
        for i in lst_sbd:
            df_final['subscriber_%s'%i] = self.return_list(i, df_prev['subscriberData'])
        return df_final

#TODO: Programar completo el search, pero se requiere permiso de CrowdTangle
class ct_search_data():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data

class ct_posts():
    def __init__(self, json_data):
        self.json_data = json_data
        self.dict_data = get_dict(json_data)
    def raw_dict(self):
        return self.dict_data
    def status(self):
        return self.dict_data['status']
    def result(self):
        return self.dict_data['result']
    def list_of_posts(self):
        post_dict = get_dict(self.result())
        return post_dict['posts']
    def get_df(self):
        df_final = pd.DataFrame()
        lst_aux = ['platformId', 'date', 'update', 'type', 'title', 'caption', 'description', 'message', 'link', 'postUrl', 'subscriberCount', 'score', ]
        for i in lst_aux:
            df_final[i] = [k.get(i) for k in self.list_of_posts()]       
        return df_final

class get_ct_data():
    def __init__(self, token):
        self.token = token
        
    def ctpost(self, ctpost):
        url_data = main_url + "/ctpost/" + ctpost + "?token=" + self.token
        json_data = get_json(url_data)    
        ctp = ctdatapost_(json_data)
        return ctp
    
    def post(self, fbpost, includeHistory = False):
        if includeHistory:
            url_data = main_url + "/post/" + fbpost + "?token=" + self.token + "&includeHistory=true"
        else:
            url_data = main_url + "/post/" + fbpost + "?token=" + self.token
        json_data = get_json(url_data)    
        ctp = ctdatapost_(json_data)
        return ctp

    def lists(self):
        url_data = main_url + "/lists" + "?token=" + self.token
        json_data = get_json(url_data)
        ctl = ct_lists(json_data)
        return ctl

    def list(self, id_, count = 10, offset_options = 0):
        url_data = main_url + "/lists/" + str(id_) + "/accounts?token=" + self.token
        #options:
        if count > 100 : count = 100
        if count == 0 : count = 1
        url_data += "&offset=%d&count=%d"%(offset_options, count)
        json_data = get_json(url_data)
        cta = ct_accounts(json_data)
        return cta

    def links(self, link, count=100, includeHistory=False, includeSummary=False, **kwargs):
        url_data = main_url + "/links" + "?token=" + self.token + "&link=" + link
        if count > 100: count = 100
        if count == 0: count = 1
        url_data +=  '&count=%d'%count
        if includeHistory:
            url_data += '&includeHistory=true'
        if includeSummary:
            url_data += '&includeSummary=true'
        for key, value in kwargs.items():
            if key == 'startDate':
                url_data += '&startDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'endDate':
                url_data += '&endDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'sortBy':
                url_data += '&sortBy=%s'%value #date, subscriber_count, total_interactions
        json_data = get_json(url_data)
        ctl = ctdatalink_(json_data)
        return ctl

    #TODO: Preguntar que datos sería útiles:

    def posts(self, count=10, includeHistory=False, includeSummary=False, **kwargs):
        url_data = main_url + "/posts" + "?token=" + self.token
        if count > 100: count = 100
        if count == 0: count = 1
        url_data +=  '&count=%d'%count
        if includeHistory:
            url_data += '&includeHistory=true'
        if includeSummary:
            url_data += '&includeSummary=true'
        lst_aux = ['weightAngry', 'weightComment', 'weightHaha', 'weightLike', 'weightLove', 'weightRepost', 'weightSad', 'weightShare', 'weightUpvote', 'weightView', 'weightWow']
        for key, value in kwargs.items():
            if key == 'startDate':
                url_data += '&startDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'endDate':
                url_data += '&endDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'language':
                url_data += '&language=%s'%value #en, es, zh-CN, zh-TW, etc.
            if key == 'sortBy':
                url_data += '&sortBy=%s'%value #overperforming, date, interaction_rate, total_interactions, underperforming
            if key == 'types':
                url_data += '&types=%s'%value #episode, extra_clip, link, live_video, live_video_complete, live_video_scheduled, native_video, photo, status, trailer, video, vine, youtube
            if key in lst_aux:
                url_data += '&%s=%d'%(key,value) #0 (default) - 10

        json_data = get_json(url_data)
        ctps = ct_posts(json_data)
        return ctps

    def leaderboard(self, count = 50, **kwargs):
        url_data = main_url + "/leaderboard" + "?token=" + self.token
        if count > 100: count = 100
        if count == 0: count = 1
        url_data +=  '&count=%d'%count
        for key, value in kwargs.items():
            if key == 'startDate':
                url_data += '&startDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'endDate':
                url_data += '&endDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'orderBy':
                url_data += '&orderBy=%s'%value #asc, desc
            if key == 'sortBy':
                url_data += '&sortBy=%s'%value #interaction_rate, total_interactions
        json_data = get_json(url_data)
        ctlb = ct_leaderboard_data(json_data)
        return ctlb

    def search(self, count = 10, includeHistory = False, **kwargs):
        url_data = main_url + "/posts/search" + "?token=" + self.token
        if count > 100: count = 100
        if count == 0: count = 1
        url_data +=  '&count=%d'%count
        if includeHistory:
            url_data += '&includeHistory=true'
        for key, value in kwargs.items():
            if key == 'startDate':
                url_data += '&startDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'endDate':
                url_data += '&endDate=%s'%value #1."yyyy-mm-ddThh:mm:ss" 2."yyyy-mm-dd"
            if key == 'orderBy':
                url_data += '&orderBy=%s'%value #asc, desc
            if key == 'sortBy':
                url_data += '&sortBy=%s'%value #interaction_rate, total_interactions
            if key == 'language':
                url_data += '&language=%s'%value #es, en, zh-CN, zh-TW, ...
            if key == 'searchField':
                url_data += '&searchField=%s'%value # text_fields_and_image_text, include_query_strings, text_fields_only , account_name_only, image_text_only 
            if key == 'searchTerm':
                url_data += '&searchTerm=%s'%value
        json_data = get_json(url_data)
        ctsc = ct_search_data(json_data)
        return ctsc


if __name__ == '__main__':

    print("Module CrowdTangle Extractor")
