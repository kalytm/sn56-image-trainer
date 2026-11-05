"""D6 integration/selection checks written by sn56-codex in review image-v1-9b6bcaf (adopted unchanged)."""
import copy
import json
import types
from unittest import mock
import pytest
from godtrainer import engine
from godtrainer.engine import Trainer, HoldoutAborted
from test_engine_deadlines import clock, member_trainer, BATCHES
from test_extra_members import make, members
from test_engine_selection import run_train_validation

@pytest.mark.parametrize('soup_vec,expected', [([.1,.2,.3,.4],'soup@200'),([.4]*4,'m0:deltaavg2@100'),(None,'m0:deltaavg2@100')])
def test_real_paired_selection_and_vector_tags(clock,tmp_path,monkeypatch,soup_vec,expected):
    t=make(clock,tmp_path)
    t.cfg.__dict__.update(select_rule='paired',select_k=1.0,deadline_ts=3000)
    t.last_holdout_unit=[.9]*4
    t._base_eval=lambda *a,**kw:.9
    t._better=types.MethodType(Trainer._better,t)
    t.evaluate_candidates=types.MethodType(Trainer.evaluate_candidates,t)
    sequence=iter([[.2,.3,.4,.5],[.199,.302,.398,.500],soup_vec])
    def score(*a,**kw):
        v=next(sequence)
        if v is None:raise HoldoutAborted(10000,0,10)
        t.last_holdout_unit=v
        return sum(v)/len(v)
    t.holdout_score=score
    monkeypatch.setattr(engine,'soup_tensors',lambda states,dev:('soup',len(states)))
    def member(g,e,mi,budget_end):
        clock.adv(500)
        tag=f'm{mi}:'+('deltaavg2' if mi==0 else 'ema')+'@100'
        scored=t.evaluate_candidates(g,e,[(tag,None,None)],100,limit=budget_end)[0][0]
        t.summary['steps']+=100
        return scored,tag,{'member':mi}
    t.train_member=member
    t.run(None,None)
    assert t.best[1]==expected
    assert t.vectors['m0:deltaavg2@100']==[.2,.3,.4,.5]
    assert t.vectors['m1:ema@100']==[.199,.302,.398,.500]
    if soup_vec is None:assert 'soup@200' not in t.vectors
    else:assert t.vectors['soup@200']==soup_vec
    assert t.saves[-1][0]==(2 if expected.startswith('soup') else 1)


def test_second_member_failure_keeps_first_saved(clock,tmp_path):
    t=make(clock,tmp_path)
    def member(g,e,mi,budget_end):
        clock.adv(500)
        return (.4,'m0:ema@100',{'m':0}) if mi==0 else None
    t.train_member=member;t.run(None,None)
    assert t.best[1]=='m0:ema@100' and t.saves[-1]==(1,'m0:ema@100') and not t.soups

@pytest.mark.parametrize('extra',[['--crossfit'],['--retrain-all'],['--extra-member-min-frac','nan'],['--extra-member-min-frac','inf']])
def test_additional_validation(tmp_path,extra):
    r=run_train_validation(['--extra-members','1','--schedule','steps','--epoch-cap','5']+extra,tmp_path)
    assert r.returncode and '--extra-' in r.stderr


def test_real_member_loop_with_cap_and_extra(clock,tmp_path,monkeypatch):
    t=member_trainer(clock,monkeypatch,prep_s=0,opt_s=.1,eval_cost=lambda i:2/BATCHES,deadline=600)
    t.cfg.__dict__.update(soup=1,retrain_all=False,crossfit=False,extra_members=1,extra_member_min_frac=.8,
                         epoch_cap=5.,eval_grid_gamma=1.,eval_count=6,delta_average=2)
    t.items=[];t.out_dir=str(tmp_path);t.summary['members']=[];t.lora.scale=1.
    t.remaining=lambda:t.cfg.deadline_ts-clock()
    monkeypatch.setattr(engine,'soup_tensors',lambda states,dev:{})
    t.run(None,None)
    schedules=t.summary['schedules']
    assert len(schedules)==2
    assert all(s['steps']==s['horizon']==1600 for s in schedules)
    assert t.summary['steps']==3200
    assert clock()<=510
    print('real run schedules',[(s['member'],s['steps'],s['horizon']) for s in schedules], 'end',clock())
