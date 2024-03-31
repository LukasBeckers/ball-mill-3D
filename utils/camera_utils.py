import cv2

from utils.video_utils import *
import json
from collections import defaultdict


class stereoCamera():
    def __init__(self, name="", **kwargs):
        """
        Possible **kwargs are all dicts with the keys = int(camera_index) ie. 0 or 1:

        camera_size
        anchor_point
        projection_error
        camera_matrix
        optimized_camera_matrix
        distortion
        """

        def recursive_defaultdict():
            return defaultdict(lambda: None)

        self.conf = defaultdict(recursive_defaultdict,
                                {key: defaultdict(lambda: None, {k: v for k, v in value.items()})
                                 for key, value in kwargs.items()})

        self.name = name

    def __str__(self):
        return self.name


    def undistort_image(self, img, cam):
        """
        Undistorts an image using the camera_matrix and the distortion values obtained by camera calibration.

        :return:                Undistorted image
        """
        img = cv2.undistort(img, self.conf["camera_matrix"][cam], self.conf["distortion"][cam], None,
                            self.conf["optimized_camera_matrix"][cam])
        return img

    def calibrate(self, images, cam, rows=8, columns=10, scaling=0.005):

        images = [self(img)[cam] for img in images]

        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefore the detectable
        # chessboard is one smaller in number of rows and columns.                   (W B)
        rows -= 1
        columns -= 1
        # termination criteria
        # If no chessboard-pattern is detected, change this... Don't ask me what to change about it!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Chessboard pixel coordinates (2D)
        imgpoints = []
        # Chessboard pixel coordinates in world-space (3D). Coordinate system defined by chessboard.
        objpoints = []

        for i, img in enumerate(images):
            img_old = np.array(img)
            factor = 4

            img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)

            try:
                _ = self.cutout_corners
            except AttributeError:
                self.cutout_corners = []

            global cutout_corners
            cutout_corners = []
            def mouse_callback(event, x, y, flags, param):
                global cutout_corners
                if event == cv2.EVENT_LBUTTONDOWN:
                    cutout_corners.append((x, y))
                    print("Left mouse button pressed!", cutout_corners)

            while len(self.cutout_corners) < 2:
                cv2.imshow(f'Camera: {cam}', img)
                cv2.setMouseCallback(f"Camera: {cam}", mouse_callback)
                cv2.waitKey(0)
                if len(cutout_corners) >=  2:
                    self.cutout_corners = cutout_corners
                    cv2.destroyWindow(f"Camera: {cam}")

            cutout_corners = self.cutout_corners
            cc_arr = np.array(cutout_corners[-2:], dtype=np.int32)

            img = img[cc_arr[:, 1].min() : cc_arr[:, 1].max(), cc_arr[:,0].min() : cc_arr[:,0].max()]
            offset = cc_arr.min(axis=0)

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            cv2.imshow("Cutout", img)
            cv2.waitKey(0)
            cv2.destroyWindow("Cutout")

            gray = img
            # localizing the chessboard corners in image-space.
            ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
            print("ret", ret)
            if ret:
                # trying to improve the detection of the chessboard corners!
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # adjusting the corner coordinates for the scaling and the cutout
                corners = np.array([(np.array(coord) + offset) / factor for (coord) in corners], dtype=np.float32)

                img = np.array(img_old)
                # resizing again to properly display the corners
                img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)
                cv2.drawChessboardCorners(img, (rows, columns), corners*factor, ret)
                for i, [corner] in enumerate(corners):
                    cv2.putText(img, f'{i}', (int(corner[0]*factor), int(corner[1])*factor), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 1)

                cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):  # press "s" to switch the ordering of the corners
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                    corners = corners[::-1]

                    # drawing the new corners
                    img = np.array(img_old)
                    img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)
                    cv2.drawChessboardCorners(img, (rows, columns), corners*factor, ret)
                    for i, [corner] in enumerate(corners):
                        cv2.putText(img, f'{i}', (int(corner[0] * factor), int(corner[1]) * factor),
                                    cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255), 1)
                    cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                objpoints.append(objp)
                imgpoints.append(corners)

        height = img.shape[0]
        width = img.shape[1]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        # saving the optimized camera matrix
        height, width = img.shape[:2]
        optimized_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1,
                                                                   (width, height))

        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', dist)

        self.conf["projection_error"][cam] = ret
        self.conf["camera_matrix"][cam] = mtx
        self.conf["optimized_camera_matrix"][cam] = optimized_camera_matrix
        self.conf["distortion"][cam] = dist

        # closing the window after calibration
        cv2.destroyAllWindows()
        return

    def stero_calibrate(self, images, rows=8, columns=10, scaling=0.005):
        """

        """
        assert self.conf["camera_matrix"][0] is not None and self.conf["camera_matrix"][1] is not None, \
            "Calibrate both cameras first!"

        def draw_lines(img):
            global line # line = (x1, y1, x2, y2)
            line = []
            lines = []
            def mouse_callback(event, x, y, flags, param):
                global line
                if event == cv2.EVENT_LBUTTONDOWN:
                    line.extend((x, y))
                    print("Current line:", line)
                if event == cv2.EVENT_RBUTTONDOWN:
                    line.pop()
                    line.pop()
                    print("Removing last point. Line:", line)


            img_old = np.array(img)
            while True:
                cv2.imshow("Drawing Lines", img)
                cv2.setMouseCallback("Drawing Lines", mouse_callback)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('r'):
                    print("Removing last line")
                    lines.pop()
                    img = np.array(img_old)
                    for l in lines:
                        cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)

                if key & 0xFF == 27:
                    print("Escaping Line drawing")
                    break
                if len(line) == 4:
                    lines.append(line)
                    line = []
                    img = np.array(img_old)
                    for l in lines:
                        cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)
                    print("Line drawn")

            return img, line

        def line_intersection(lines):
            """Finds the intersection of two lines given in Hesse normal form."""

            intersections = []
            for i, line1 in enumerate(lines):
                for line2 in lines[i + 1:]:

                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2

                    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
                    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)


                    intersections.append((px, py))
            return np.array(intersections)

        # open cv can only detect inner corners, so reducing the rows and columns by one
        rows -= 1
        columns -= 1

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Pixel coordinates in image space of checkerboard
        imgpoints_1 = []
        imgpoints_2 = []

        # World coordinates of the checkerboard. (World coordinate system has its origin in the bottom left corner of
        # the checkerboard.
        objpoints = []

        for img in images:
            img1, img2 = self(img)

            assert img1.shape == img2.shape, "For both cameras must have the same resolution for stereo-calibration"

            height, width = img1.shape[:2]

            factor = 4

            img1 = cv2.resize(img1, np.array(img1.shape[:2])[::-1] * factor)
            img2 = cv2.resize(img2, np.array(img2.shape[:2])[::-1] * factor)
            img1_old = np.array(img1)
            img2_old = np.array(img2)
            #img1, lines1 = draw_lines(img1)
            #img2, lines2 = draw_lines(img2)

            #corners1 = line_intersection(lines1)
            #corners2 = line_intersection(lines2)
            corners1 = np.array([(345.8016528924191, -11140.603305780522), (377.2376992377254, -3595.9521829524324), (
            381.23697011812436, -2636.1271716468855), (377.0812217979999, -3633.5067684832916), (
            374.4725697786213, -4259.583253127871), (377.34104581213273, -3571.1490050903185), (
            379.2387152777712, -3115.7083333332794), (378.7924644842554, -3222.808523780161), (
            393.8587279275379, 393.09470260918715), (393.7291880652357, 362.00513565648754), (
            393.59154690024627, 328.9712560591884), (393.4646852137902, 298.5244513095584), (
            393.34209980288375, 269.10395269218776), (393.2298357568787, 242.1605816508005), (
            393.1086367979612, 213.0728315107759), (421.9127516778169, -2083.3825503353955), (
            423.94942528737255, -1841.0183908046683), (414.69801223240006, -2941.9365443423953), (
            408.6036542515955, -3667.1651440619703), (413.18339100344997, -3122.1764705881583), (
            416.22120658136043, -2760.6764168190634), (414.9883419689052, -2907.387305699435), (
            442.7325023486784, 394.1677794926859), (442.46961530687827, 362.88422151856065), (
            442.19296740994895, 329.9631217838768), (441.94039202259654, 299.9066506890351), (
            441.6916611446312, 270.30767621106986), (441.46530684016136, 243.3715139792507), (
            441.22099684137305, 214.29862412334728), (435.3266041816557, -1629.2278298484766), (
            374.99644886366303, -3671.8345170457155), (339.9264069263859, -4859.205936919922), (
            378.3397955587021, -3558.6383503702345), (396.24381625440765, -2952.459363957528), (
            391.53977272728065, -3111.7248376624007), (495.1233187257707, 395.31807685825015), (
            494.1929103214894, 363.8171065989851), (493.2237635705665, 331.00456660348067), (
            492.3477109117654, 301.34392658404244), (491.46763031099385, 271.5469119579498), (
            490.67193159131506, 244.6068267345102), (489.81331863048285, 215.53664506064905), (
            -3609.666666657411, -76928.33333313608), (-815.7074235811422, -24917.707423591666), (
            235.69414893615453, -5345.4627659570915), (330.1689419795335, -3586.7781569967096), (
            324.13429522751596, -3699.115427302894), (544.1410168539794, 396.39431374330127), (
            542.4377450452164, 364.6872539186509), (540.6803714647535, 331.9730688054034), (
            539.1066250842881, 302.6771746459876), (537.4958965730381, 272.69284389808416), (
            536.048337950138, 245.74598337950115), (534.4866638553586, 216.6748194612832), (
            -188.18716577530046, -14025.748663094104), (387.1700069108768, -3448.02833448538), (
            433.66725043781315, -2593.194395796757), (418.0090909091028, -2881.063636363718), (
            596.343205458376, 397.5404695809221), (594.6073910562014, 365.62818941784764), (
            592.8346725936052, 333.0374422978284), (591.264169945463, 304.16435515119474), (
            589.6229203247505, 273.99061212426744), (588.1577581824243, 247.05416966148366), (
            586.5775132769336, 218.00197486055865), (509.5431906614389, -1915.1431906613295), (
            516.0349907919159, -1802.4640883978566), (490.07438016527476, -2253.066115702413), (
            642.8430450122961, 398.5614241419869), (640.993857889392, 366.4648190801691), (
            639.1224249558569, 333.98209030522185), (637.4804519925782, 305.48213101404474), (
            635.7322628635926, 275.1385625609195), (634.1807976824783, 248.20955977445533), (
            632.5078619241376, 219.17217482609286), (526.7517985611132, -1699.5827338128272), (
            465.4023668639283, -2468.0650887575184), (694.3409673272894, 399.69211704710585), (
            691.7614658785243, 367.38046732045467), (689.1767585673789, 335.0036073177013), (
            686.9325682722663, 306.8921709894335), (684.4945321780442, 276.352560967088), (
            682.3443489289747, 249.41868658399122), (680.0263609546789, 220.38283722177508), (
            262.70967741931247, -4234.38709677351), (745.5433668153217, 400.81632142708327), (
            742.1551428451911, 368.28937131384083), (738.7933532168732, 336.0161908819773), (
            735.9050557494248, 308.2885351944844), (732.7034148032594, 277.55278211128467), (
            729.8971333682916, 250.61248033560545), (726.8725385843541, 221.57637040979264), (
            794.7422254297329, 401.89653588767845), (790.9867329470642, 369.17010139583914), (
            787.2957746478867, 337.0060362173035), (784.1582085552548, 309.66438883864294), (
            780.6101371171287, 278.7454805921275), (777.5189516406418, 251.8080071541586), (
            774.1880825217294, 222.7818619750757), (-7536.5255102033125, 218.974489795896), (
            -41028.052631589744, -516.3684210527674), (14814.606943584304, 709.7179169250285), (
            42560.65915492358, 1318.9126760561521), (48327.042440324716, 1445.519893899396), (
            51507.494584831344, 1515.3501805052326), (14318.603448278333, 613.1551724138989), (
            6451.96493961798, 471.271912738587), (13934.389090909935, 606.2254545454913), (
            17346.61921708082, 667.7686832739818), (20404.98970840597, 722.9296740995268), (
            4149.6564102566235, 405.6256410256618), (13731.320754715685, 601.1698113206979), (
            18876.018181819898, 706.1636363637006), (23252.871794869807, 795.4871794871115), (
            -7739.703271027132, 66.62149532709502), (-16140.415000002018, -172.91000000002163), (
            -27754.012820508866, -504.0512820512102), (129705.75000027023, 3488.5000000072678), (
            96748.99999992672, 2667.999999997979), (78346.35714295041, 2199.1428571454753)])
            corners2 = np.array([(615.0578034681191, -6141.6242774557595), (731.5185185185939, -8742.58024691448), (
            456.55042016805766, -2601.626050420113), (508.6597659766091, -3765.401440144099), (
            508.36496350364035, -3758.8175182481064), (516.8471781864381, -3948.2536461636646), (
            482.98262032085023, -3191.9451871657398), (516.676762896362, -3944.447704685328), (
            336.71874171131526, 74.61476844729727), (335.069137487532, 111.45592944511516), (
            333.5654913492086, 145.03735986767842), (332.0126889152523, 179.71661422602764), (
            330.4323542839752, 215.01075432455752), (328.5710413372485, 256.58007680144556), (
            326.7222167341699, 297.8704929368766), (939.2268041233887, -15250.773195871048), (
            467.29142185665023, -1989.3889541716096), (514.898322318238, -3327.1428571427728), (
            518.1543357199785, -3418.6368337311737), (527.4223391614238, -3679.067730436196), (
            501.3859164653147, -2947.444252675215), (529.9207519351206, -3749.273129377027), (
            393.7815184524485, 76.23933148619066), (392.4779026967476, 112.87093422139877), (
            391.2934046256525, 146.1553300191587), (390.0418875862665, 181.3229588259175), (
            388.8077034743054, 216.00353237201188), (387.31091078867996, 258.0634068380981), (
            385.86641068526046, 298.65385974417546), (484.56603773583686, -1004.7358490565784), (
            528.4044298605559, -2378.33880229703), (538.7566204287402, -2702.707440100826), (
            550.869263607267, -3082.2369263607807), (533.1141942369201, -2525.9114194236627), (
            558.6469916222462, -3325.939070830202), (450.01573492699924, 77.84030562069752), (
            448.8533892278322, 114.26047086124933), (447.8005428170335, 147.24965839962573), (
            446.66307465980117, 182.89032732622277), (445.5754583921018, 216.96897038081818), (
            444.2180705427603, 259.50045632683725), (442.94436619718334, 299.40985915492973), (
            820.6603773588002, 18152.64150944081), (323.6179775281263, -10178.775280900021), (
            365.207492795368, -7808.172910662374), (435.01307189543434, -3829.2549019608678), (
            403.8119551681095, -5607.7185554170455), (503.585358235992, 79.36541945155845), (
            504.22149434522, 115.62517767752308), (504.7956743395641, 148.35343735516753), (
            505.4301245065293, 184.51709687215316), (506.01748940488244, 217.99689607831422), (
            506.77333569719025, 261.08013473982817), (507.46077861708284, 300.26438117373607), (
            509.1338028168566, -3732.1003521123475), (499.92541856927954, -4378.9893455101155), (
            524.7153605015565, -2637.4959247648353), (503.3062337046813, -4141.487082247077), (
            563.4137896083826, 81.06871998885079), (563.9266446048933, 117.09678349378252), (
            564.3880073199472, 149.50751422626666), (564.9098020375227, 186.1635931359867), (
            565.3773155435023, 219.00641693101204), (565.997518788881, 262.575694918911), (
            566.5451523937489, 301.0469556608445), (464.7579462102121, -5274.161369192509), (
            550.4858569051734, -2295.1164725458207), (492.96909927677535, -4293.823800131351), (
            618.9107540933196, 82.64870474287028), (619.9417967957821, 118.47743865341722), (
            620.8662239570128, 150.60128250620969), (621.9350257853961, 187.7421460425024), (
            622.8628507075352, 219.98406208686274), (624.1307541269729, 264.0437059122974), (
            625.2179591058574, 301.82407892855423), (615.9711846317817, -1425.0971184631296), (
            517.4366071428802, -3933.2500000001755), (675.2670882532577, 84.25315553745934), (
            676.6665407902518, 119.87558375187234), (677.9170269130818, 151.70613960571117), (
            679.3952147664288, 189.3327395091052), (680.637974837416, 220.96663222512623), (
            682.388083416087, 265.5148505913152), (683.845023427517, 302.60059633678844), (
            1343.7253886012684, 8243.637305700906), (729.6948268589174, 85.802699697051), (
            732.3627811705804, 121.24837840913408), (734.7381272868595, 152.80654823971022), (
            737.6087021363861, 190.94418552626684), (739.944373237631, 221.97524444281675), (
            743.3373959410201, 267.05397464497537), (746.074987525839, 303.4248342718653), (
            790.2750738590062, 87.52740423797886), (792.6641389648174, 122.73467948152714), (
            794.7836379548078, 153.9694014392657), (797.4049623613308, 192.59944532488103), (
            799.4670112496203, 222.98753420492562), (802.5587135242565, 268.54946246273363), (
            804.9781970550641, 304.20500923251757), (9986.937704917214, 349.35409836062706), (
            8079.152099105578, 295.03991741226065), (133864.76562494773, 3876.124999998486), (
            12593.85322069727, 423.57233368533326), (56960.36871507849, 1686.675977653474), (
            15009.445820433728, 492.3436532507835), (6698.666666666273, 268.3051643192331), (
            -22193.076305218656, -443.8152610441321), (13897.761755485348, 445.74764890280386), (
            -239954.58823517652, -5811.176470585387), (16692.826175868693, 514.6400817995778), (
            -3842.0571428572835, 64.17142857143092), (30013.502538074878, 719.8248730965381), (
            -18637.317220544744, -222.35649546828915), (25316.358095239302, 628.8590476190776), (
            3640.8169790517186, 271.309812568901), (32010.1007194283, 1056.618705036098), (
            8521.179365079195, 406.40634920634113), (-4716.587500000164, 129.17708333333783), (
            22370.293413175328, 589.8383233533376), (3769.3314763230323, 343.46796657380816)])

            # Ignoring intersections outside the image
            w, h = img1.shape[:2]
            print(w, h)
            corners1 = np.array([[corner] for corner in corners1 if h > corner[0] > 0 and w > corner[1] > 0])
            corners2 = np.array([[corner] for corner in corners2 if h > corner[0] > 0 and w > corner[1] > 0])

            print("Corners1", corners1, len(corners1))
            print("Corners2", corners2, len(corners2))

            for i, [corner] in enumerate(corners1):
                cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
            for i, [corner] in enumerate(corners2):
                cv2.putText(img2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
            cv2.imshow(f'Detection 1', img1)
            cv2.imshow(f'Detection 2', img2)
            key = cv2.waitKey(0)

            if key & 0xFF == ord('s'):  # press s to switch ordering of img1
                cv2.destroyWindow(f'Detection 1')
                print("Corners1 before", corners1.shape)
                corners1 = corners1[::-1]
                print("Corners1 after", corners1.shape)
                # drawing the new corners
                img1 = np.array(img1_old)
                for i, [corner] in enumerate(corners1):
                    cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (0, 0, 255), 1)
                cv2.imshow(f'Detection 1', img1)
                cv2.waitKey(0)

            # adjusting corner coordinates for scaling
            corners1 /= factor
            corners2 /= factor
            objpoints.append(objp)

            imgpoints_1.append(corners1)
            imgpoints_2.append(corners2)
            cv2.destroyWindow(f'Detection 1')
            cv2.destroyWindow(f'Detection 2')
        # prerform stereo calibration on accumulated objectpoints
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

        imgpoints_1 = np.array(imgpoints_1, dtype=np.float32)
        imgpoints_1 = np.swapaxes(imgpoints_1, axis1=1, axis2=2)
        imgpoints_2 = np.array(imgpoints_2, dtype=np.float32)
        imgpoints_2 = np.swapaxes(imgpoints_2, axis1=1, axis2=2)
        objpoints = np.array(objpoints)
        objpoints = np.expand_dims(objpoints, axis=1)
        print("objectpoints shape", objpoints.shape)
        print("imgpoints shape", imgpoints_1.shape)
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints_1,
                                                                      imgpoints_2,
                                                                      self.conf["camera_matrix"][0],
                                                                      self.conf["distortion"][0],
                                                                      self.conf["camera_matrix"][1],
                                                                      self.conf["distortion"][1],
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        # Matrix that rotates the coordinate system of the second camera to match the first.
        self.conf["rotation_matrix"][0] = R
        # Matrix that translates the coordinate system of the second camera to match the first.
        self.conf["translation_matrix"][0] = T
        self.conf["stereo_calibration_error"][0] = ret


        print(f'Stereo-calibration error: {ret}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')


        cv2.destroyAllWindows()
        return

    def set_anchor_point(self, img, cam):
        """
        img: frame from Video
        cam: number of camera for which to set the anchor-point (0 or 1)
        """
        def mouse_callback(event, x, y, flags, param):
            global anchor_point
            if event == cv2.EVENT_LBUTTONDOWN:
                anchor_point = (x, y)
                print("Left mouse button pressed!", anchor_point)

        win_name = "Set Anchor Point" + str(cam)
        cv2.imshow(win_name, img)
        cv2.setMouseCallback(win_name, mouse_callback)
        cv2.waitKey(0)
        self.conf[f"anchor_points"][cam] = anchor_point
        cv2.destroyWindow(win_name)

    def draw_camera_region(self, img):

        for anchor_point in self.conf["anchor_point"].values():
            # Drawing camera
            start_point = anchor_point - np.array(self.conf["camera_size"][0])/2
            end_point = anchor_point + np.array(self.conf["camera_size"][0])/2
            img = cv2.rectangle(img,  start_point.astype(np.int32), end_point.astype(np.int32), (255, 0, 0), 5)

        return img

    def __call__(self, image):

        # Camera0
        anchor_point0 = np.array(self.conf["anchor_point"][0])
        start_point0 = anchor_point0 - np.array(self.conf["camera_size"][0]) / 2
        end_point0 = anchor_point0 + np.array(self.conf["camera_size"][0]) / 2

        # Camera 1
        anchor_point1 = np.array(self.conf["anchor_point"][1])
        start_point1 = anchor_point1 - np.array(self.conf["camera_size"][0]) / 2
        end_point1 = anchor_point1 + np.array(self.conf["camera_size"][0]) / 2

        # checking for negative values and adjusting the anchor size
        for i, val in enumerate(start_point0):
            if val < 0:
                self.anchor_points[0][i] -= val
                return self(image)

        for i, val in enumerate(start_point1):
            if val < 0:
                self.anchor_points[1][i] -= val
                return self(image)

        frame0 = image[int(start_point0[1]): int(end_point0[1]), int(start_point0[0]): int(end_point0[0])]
        frame1 = image[int(start_point1[1]): int(end_point1[1]), int(start_point1[0]): int(end_point1[0])]

        return frame0, frame1

if __name__=="__main__":

    sC = stereoCamera(camera_size={0:(300, 150), 1:(300, 150)},
                      anchor_point={0:(587, 269), 1:(598, 433)},
                      camera_matrix={0:np.array([[2.24579312e+03, 0.00000000e+00, 6.06766474e+02],
                                                 [0.00000000e+00, 3.18225724e+03, 2.87228912e+02],
                                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                                    1:np.array([[9.17450924e+02, 0.00000000e+00, 5.97492459e+02],
                                                [0.00000000e+00, 1.08858369e+03, 2.96145751e+02],
                                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])},
                      optimized_camera_matrix={0:np.array( [[1.98885152e+03, 0.00000000e+00, 5.83904948e+02],
                                                           [0.00000000e+00, 2.71756632e+03, 3.41261625e+02],
                                                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                                               1:np.array([[9.35319179e+02, 0.00000000e+00, 5.90025655e+02],
                                                          [0.00000000e+00, 1.09136910e+03, 2.97696817e+02],
                                                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])},
                      projection_error={0: 0.26768362133770185, 1: 0.29408707559840946},
                      distortion={
                          0:np.array([[-1.53486495e+00,  1.95803727e+01,  1.63594781e-01, -2.81574724e-02, -1.10093707e+02]]),
                          1:np.array([[ 0.03667417,  0.10305058,  0.00557331, -0.00655738,-0.16404791]])},
                      stereo_calibration_error= {0: 0.6988643727550614},
                      translation_matrix={0: [[-0.60330682], [-0.39384531], [1.07405106]]},
                      rotation_matrix={0: [[0.73971458,  0.1145444,   0.66310023],
                                          [-0.09028238, - 0.95960383,  0.26647622],
                                          [0.66683688, - 0.25698261, - 0.69949161]]}
                      )
    vL = videoLoader()
    vL.load_video("../videos/WhatsApp Video 2024-03-29 at 19.14.15 (2).mp4", start_frame=100, end_frame=-100)
    #frame = vL[10]
    #sC.set_anchor_point(frame, 0)
    #sC.set_anchor_point(frame, 1)
    frames = vL[:2]

    for frame in frames[:1]:
        frame = sC.draw_camera_region(frame)
        cv2.imshow("Frame", frame)
        frame0, frame1 = sC(frame)
        cv2.imshow("frame0", frame0)
        cv2.imshow("frame1", frame1)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    #sC.calibrate(frames, 0)
    #sC.calibrate(frames, 1)
    sC.stero_calibrate(frames)