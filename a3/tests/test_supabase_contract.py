"""Offline adapter-contract tests, not a substitute for a hosted Supabase pilot."""
from storage import SupabaseStore


class Result:
    def __init__(self,data):self.data=data


class Query:
    def __init__(self,client,table):
        self.client,self.name=client,table
        self.filters={};self.bounds=None;self.payload=None
    def select(self,*args):return self
    def eq(self,key,value):self.filters[key]=value;return self
    def order(self,key):self.order_key=key;return self
    def range(self,start,end):self.bounds=(start,end);return self
    def limit(self,n):self.bounds=(0,n-1);return self
    def upsert(self,row,on_conflict):
        self.client.writes.append((self.name,row,on_conflict));self.payload=row
        return self
    def execute(self):
        if self.payload is not None:return Result([self.payload])
        rows=[r for r in self.client.data.get(self.name,[]) if all(r.get(k)==v for k,v in self.filters.items())]
        if self.bounds:
            start,end=self.bounds;rows=rows[start:end+1]
        return Result(rows)


class Client:
    def __init__(self):self.data={};self.writes=[];self.calls=[]
    def table(self,name):return Query(self,name)
    def rpc(self,name,payload):
        self.calls.append((name,payload))
        class RPC:
            def execute(self):return Result(None)
        return RPC()


def adapter():
    store=SupabaseStore.__new__(SupabaseStore);store.client=Client()
    return store


def test_paginated_history_and_conflict_keys():
    store=adapter()
    store.client.data['trials']=[{'id':i,'participant_id':'person','is_pilot':False} for i in range(1201)]
    assert len(store.rows('trials',participant_id='person',is_pilot=False))==1201
    assert store.rows('trials',participant_id='missing')==[]
    payload={'session_id':'uuid','trial_idx':1,'ai_probs':[.125]*8}
    store.upsert('trials',payload)
    assert store.client.writes==[('trials',payload,'session_id,trial_idx')]
    store.health()


def test_start_rpc_is_idempotent():
    store=adapter();payload={'session_id':'uuid'}
    store.start(payload)
    assert store.client.calls==[('start_study_session',{'payload':payload})]
    store.client.data['sessions']=[payload]
    store.start(payload)
    assert len(store.client.calls)==1
