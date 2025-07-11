from rational_linkages import (
    DualQuaternion,
    MotionFactorization,
    Plotter,
    PointHomogeneous,
    RationalMechanism,
    TransfMatrix,
)

# testing code, for examples see 'examples' folder or read
# the documentation/docstrings of classes

if __name__ == '__main__':
    f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                              DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

    f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                              DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

    # h1 = DualQuaternion([-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5])
    # h2 = DualQuaternion([-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480, -1609 / 720, -1217 / 300])
    # h3 = DualQuaternion([9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77, -27 / 70])
    # k1 = DualQuaternion([9 / 4, -353293129020116088274366524 / 2046064697881244081606857985, 71057440088136127923615537 / 292294956840177725943836855,-770841181127162033209449696 / 3215244525241954985382205405,0,1285925291840670577611498917452530201753024451488061/ 3106776065243685802900593279235046876397736685683310,369908388252939453727221870302615264755351380922518/ 10873716228352900310152076477322664067392078399891585,-521169721602430808498118610255124429182751294514817/ 1977039314245981874573104814058666194071286981798470])
    # k2 = DualQuaternion([-3 / 10, -1584906194063534950985110611269499455843766362303 / 748589572249221191178253511800771099480289042160, -1920849060977583492916015219909062271715671063459 / 499059714832814127452169007867180732986859361440,-86070495207833111808000603712957119046286563789/ 74858957224922119117825351180077109948028904216,0,-1493945822997033838889866124648869571241092592593232647136782680036917092967587145383237343823/ 933977246133786589561715243908633265522422939574517641472244621670941695011531052831617095776,-4383675369926674968359098051148186009275012265484280981512415092271834593351409097148552548873/ 7004829346003399421712864329314749491418172046808882311041834662532062712586482896237128218320,4903731966877692899272930689015951377076396084948182642657351735217230221511376454058771819441/ 972892964722694364126786712404826318252523895390122543200254814240564265637011513366267808100])
    # k3 = DualQuaternion([-1 / 4, 71409809286507251213549 / 8803698436759977863535, -24700620508978006448565 / 586913229117331857569,-1322302914772604754264 / 586913229117331857569,0,-6009699344068792249739386119282475891686296/ 344467138512933679794546525402592102589761,-50778766896365492707380727887969151002044936/ 15501021233082015590754593643116644616539245,-8366061133662094762021335484888717836156016/ 5167007077694005196918197881038881538846415])
    #
    # h1 = DualQuaternion([0, 1, 0, 0, 0, 0, 0, 0])
    # h2 = DualQuaternion([0, 0, 3, 0, 0, 0, 0, 1])
    # h3 = DualQuaternion([0, 1, 1, 0, 0, 0, 0, -2])
    # k1 = DualQuaternion([0, 47/37, 23/37, 0, 0, 0, 0, 24/37])
    # k2 = DualQuaternion([0, -93/481, 1440/481, 0, 0, 0, 0, -164/481])
    # k3 = DualQuaternion([0, 12/13, 5/13, 0, 0, 0, 0, -17/13])
    #
    # #f1 = MotionFactorization([h1, h2, h3])
    # #f2 = MotionFactorization([k1, k2, k3])
    # #"""
    # f1.set_joint_connection_points([PointHomogeneous([1, 0, 0, 0.1]),
    #                                 PointHomogeneous([1, 0, 0, 0.5]),
    #                                 PointHomogeneous([1, -0.5, 0, 0.2]),
    #                                 PointHomogeneous([1, -0.5, 0, 0.3])])
    # f2.set_joint_connection_points([PointHomogeneous([1, -0.16666667, 0, 0]),
    #                                 PointHomogeneous([1, -0.16666667, 0, -0.1]),
    #                                 PointHomogeneous([1, -0.66666667, 0, 0.1]),
    #                                 PointHomogeneous([1, -0.66666667, 0, 0])])
    #"""
    t = TransfMatrix.from_rpy_xyz([90, 0, 45], [-0.2, 0.5, 0], unit='deg')
    tdq = DualQuaternion(t.matrix2dq())

    m = RationalMechanism([f1, f2], tool=tdq)
    #res = m.collision_check(parallel=False)
    #print(res)

    p = Plotter(mechanism=m, steps=200, arrows_length=0.2)
    p.plot(m.get_motion_curve(), label='motion curve', interval='closed', color='red', linewidth='0.7', linestyle=':')
    p.show()


    """
    
    h1li = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    
    h2li = np.array([0, 0, 1, 0, 0, 1, 0, 1])
    h2li = h2li * (4/5)
    h2li[0] = 1
    
    h3li = np.array([0, 0, 3/5, 4/5, 0, 4/5, 0, 0])
    h3li = h3li * (5/6)
    h3li[0] = 2
    
    k1li = np.array([0, -623/1689, -3496/8445, -7028/8445, 0, -3151184/14263605, 12303452/71318025, 863236/71318025])
    k1li = k1li * (-1)
    
    k2li = np.array([0, -159238240/172002693, -36875632/172002693, -53556485/172002693, 0, 4263140176797785/29584926399252249, 8149138391852807/29584926399252249, -91432397690177392/147924631996261245])
    k2li = k2li * (-4/5)
    k2li[0] = 1
    
    k3li = np.array([0, 13380/101837, -2182923/2545925, 1266764/2545925, 0, -84689025844/51853872845, -611161964208/1296346821125, -494099555856/1296346821125])
    k3li = k3li * (-5/6)
    k3li[0] = 2
    
    h1 = DualQuaternion(h1li)
    h2 = DualQuaternion(h2li)
    h3 = DualQuaternion(h3li)
    k1 = DualQuaternion(k1li)
    k2 = DualQuaternion(k2li)
    k3 = DualQuaternion(k3li)
    
    f1 = MotionFactorization([h1, h2, h3])
    f2 = MotionFactorization([k3, k2, k1])
    
    f1.set_joint_connection_points([PointHomogeneous([1, -0.72533812018960216974, 0., 0.]),
                                    PointHomogeneous([1, -0.79822634381283099450, 0., 0.]),
                                    PointHomogeneous([1, -1., 0.5585449951, 1.000000000]),
                                    PointHomogeneous([1, -1., 0.4856567714, 1.000000000]),
                                    PointHomogeneous([1, 0.0, -0.2092444750, 1.054340700]),
                                    PointHomogeneous([1, 0.0, -0.2529774091, 0.9960301212])])
    
    f2.set_joint_connection_points([PointHomogeneous([1, -0.67209203533440663286, 1.4850577244688201736, 1.0430280533405744484]),
                                    PointHomogeneous([1, -0.68166855891698272676, 1.5475534302638807256, 1.0067614006662592824]),
                                    PointHomogeneous([1, -0.6463533229, 0.5179684833, 0.0801412885]),
                                    PointHomogeneous([1, -0.5788741910, 0.5335949788, 0.1028364974]),
                                    PointHomogeneous([1, -0.1404563036, -0.1904506672, 0.1508073794]),
                                    PointHomogeneous([1, -0.1135709493, -0.1602769278, 0.2114655719])])
    
    m = RationalMechanism([f1, f2])
    p = Plotter(mechanism=m, show_tool=False, steps=500)
    
    """
