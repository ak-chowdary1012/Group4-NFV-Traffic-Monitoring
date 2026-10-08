import unittest
import numpy as np
from nfv_model import Model,Config,BYTES_PER_MBPS_SECOND,workload,calibrated

class ModelTests(unittest.TestCase):
    def constant(self,rate=8,duration=20,**kw):
        x=np.full((duration,1),float(rate));return Model(x,np.zeros_like(x,dtype=bool),15,Config(duration=duration,**kw))
    def test_byte_units_and_conservation(self):
        m=self.constant(policy=False);s=m.run();self.assertAlmostEqual(s['offered_MB'],20);self.assertAlmostEqual(s['forwarded_MB'],20)
    def test_capacity_drop(self):
        s=self.constant(rate=80,capacity=20,policy=False).run();self.assertAlmostEqual(s['mean_forwarded_Mbps'],20);self.assertAlmostEqual(s['congestion_drop_MB'],150)
    def test_proportional_share(self):
        x=np.tile([10.,30.],(10,1));m=Model(x,np.zeros_like(x,dtype=bool),100,Config(duration=10,capacity=20,policy=False));s=m.run();self.assertAlmostEqual(s['forwarded_MB'],25)
    def test_weighted_counter(self):
        x=np.array([[4.],[12.]]);m=Model(x,np.zeros_like(x,dtype=bool),100,Config(duration=2,poll=2,policy=False));s=m.run();self.assertAlmostEqual(m.polls[0]['rate'][0],8);self.assertLess(s['counter_RMSE_Mbps'],1e-10)
    def test_exact_policy_timing(self):
        m=self.constant(rate=20,poll=2,telemetry_min=.1,telemetry_max=.1,processing=.2,installation=.3,block_ttl=100);s=m.run();self.assertAlmostEqual(m.commands[0]['time'],2.6);self.assertAlmostEqual(s['policy_drop_MB'],17.4*20/8)
    def test_strict_threshold(self):
        m=self.constant(rate=15,poll=1);s=m.run();self.assertEqual(s['command_count'],0)
    def test_expiry_releases_traffic(self):
        m=self.constant(rate=20,poll=10,telemetry_min=0,telemetry_max=0,processing=0,installation=0,block_ttl=2);s=m.run();self.assertAlmostEqual(s['policy_drop_MB'],5)
    def test_total_message_loss(self):
        s=self.constant(rate=20,poll=1,telemetry_loss=1).run();self.assertEqual(s['received'],0);self.assertEqual(s['command_count'],0)
    def test_heartbeat_phase_recovery(self):
        m=self.constant(policy=False,failure=7.5);s=m.run();self.assertAlmostEqual(s['outage_s'],4.9140625);self.assertAlmostEqual(s['outage_drop_MB'],4.9140625)
    def test_no_backup(self):
        s=self.constant(policy=False,failure=7.5,backup=False).run();self.assertAlmostEqual(s['outage_s'],12.5);self.assertAlmostEqual(s['outage_drop_MB'],12.5)
    def test_reset_epoch(self):
        m=self.constant(policy=False,poll=1,failure=7.5);s=m.run();self.assertLess(s['counter_RMSE_Mbps'],1e-10);self.assertTrue(all(min(p['rate'])>=0 for p in m.polls));self.assertTrue(any(p['epoch']==1 for p in m.polls))
    def test_stale_command(self):
        m=self.constant(rate=20,poll=2,failure=2.05,telemetry_min=0,telemetry_max=0,processing=0,installation=.1);s=m.run();self.assertFalse(any(2.05<=c['time']<m.recovery for c in m.commands))
    def test_repeatability(self):
        x,l=workload(3);c=Config(failure=350.4,telemetry_loss=.1);a=Model(x,l,calibrated(3)['Fixed 15'],c,3).run();b=Model(x,l,15,c,3).run();self.assertEqual(a,b)
    def test_randomized_conservation(self):
        for seed in range(5):
            x,l=workload(seed)
            for poll in [1,10,30,60]:
                s=Model(x,l,15,Config(poll=poll,capacity=30,failure=350.1+seed*.1,telemetry_loss=.1),seed).run()
                self.assertAlmostEqual(s['offered_MB'],sum(s[k] for k in ['forwarded_MB','policy_drop_MB','outage_drop_MB','congestion_drop_MB']),places=7)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):self.constant(poll=0)
        with self.assertRaises(ValueError):self.constant(telemetry_loss=2)
if __name__=='__main__':unittest.main(verbosity=2)
