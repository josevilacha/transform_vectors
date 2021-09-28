from manim import *
from manim.mobject.geometry import ArrowTriangleTip

import numpy as np

import itertools

from copy import copy

# config.background_color = "#363538"
config.background_color = "#2E3047"
myTemplate = TexTemplate()
myTemplate.add_to_preamble(
    r"\usepackage{fourier} \usepackage[T1]{fontenc} \usepackage{bm} \usepackage{mathtools}"
)
config.tex_template = myTemplate
config.text_color = "#F6F6F6"
config.pixel_height = 1080  #
config.pixel_width = 1920  #
config.frame_width = 18
BLUE = "#707793"
ORANGE = "#F58F7C"
RED = "#E9322E"
plate_color = "#707793"
DARK_BLUE = "#348feb"

std_line_buff = 0.5
std_wait_time = 3


class VectorScene(Scene):
    def setup(self):
        self.lin_trans = np.array([[3, 0.3], [0.3, 4]]) / 4
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.lin_trans)
        self.diag_lin_trans = np.array([[self.eig_vals[0], 0], [0, self.eig_vals[1]]])
        self.rot_mat = self.eig_vecs.T
        self.b_trans = np.array([0.8, 1.6])
        self.sol_trans = self.b_trans / self.eig_vals

    def show_linear_transform(self):
        def to_global_coord(point):
            return np.array(point) + self.plane_origin_loc

        lin_trans = self.lin_trans
        self.plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-4, 4, 1], x_length=5, y_length=5
        )
        plane = self.plane

        plane.move_to(self.frame_lin_trans.get_center())
        self.plane_origin_loc = self.frame_lin_trans.get_center()

        unit_vector_x = Arrow(
            self.plane_origin_loc,
            np.array([1, 0, 0]) + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=BLUE,
        )
        unit_vector_y = Arrow(
            self.plane_origin_loc,
            np.array([0, 1, 0]) + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=BLUE,
        )

        v_1 = MathTex(r"\hat{\bm e}_1", color=BLUE)
        v_2 = MathTex(r"\hat{\bm e}_2", color=BLUE)
        v_1.next_to(unit_vector_x, DOWN)
        v_2.next_to(unit_vector_y, LEFT)

        self.play(
            DrawBorderThenFill(plane),
            GrowArrow(unit_vector_x),
            GrowArrow(unit_vector_y),
            Write(v_1),
            Write(v_2),
            run_time=2,
        )
        self.wait(2)

        self.play(
            plane.animate.apply_matrix(lin_trans, about_point=self.plane_origin_loc),
            unit_vector_x.animate.put_start_and_end_on(
                self.plane_origin_loc,
                to_global_coord(np.append(lin_trans.dot([1, 0]), 0)),
            ),
            unit_vector_y.animate.put_start_and_end_on(
                self.plane_origin_loc,
                to_global_coord(np.append(lin_trans.dot([0, 1]), 0)),
            ),
        )

        self.wait(2)
        self.play(
            plane.animate.apply_matrix(
                np.linalg.inv(lin_trans), about_point=self.plane_origin_loc
            ),
            Uncreate(unit_vector_x),
            Uncreate(unit_vector_y),
            Uncreate(v_1),
            Uncreate(v_2),
        )

    def show_linear_transform_new_coords(self):
        def to_global_coord(point):
            return np.array(point) + self.plane_origin_loc

        lin_trans = self.diag_lin_trans
        self.plane_new_coords = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-4, 4, 1], x_length=5, y_length=5
        )
        plane = self.plane_new_coords

        plane.move_to(self.frame_text.get_center())
        self.plane_new_coords_origin_loc = (
            plane_origin_loc
        ) = self.frame_text.get_center()

        # unit_vector_x = Arrow(
        #     self.plane_origin_loc,
        #     np.array([1, 0, 0]) + self.plane_origin_loc,
        #     buff=0,
        #     stroke_width=2,
        #     max_tip_length_to_length_ratio=10,
        #     max_stroke_width_to_length_ratio=20,
        #     color=BLUE,
        # )
        # unit_vector_y = Arrow(
        #     self.plane_origin_loc,
        #     np.array([0, 1, 0]) + self.plane_origin_loc,
        #     buff=0,
        #     stroke_width=2,
        #     max_tip_length_to_length_ratio=10,
        #     max_stroke_width_to_length_ratio=20,
        #     color=BLUE,
        # )

        v_1 = MathTex(r"\hat{\bm e}_1", color=BLUE)
        v_2 = MathTex(r"\hat{\bm e}_2", color=BLUE)
        # v_1.next_to(unit_vector_x, DOWN)
        # v_2.next_to(unit_vector_y, LEFT)

        local_eig_vec_1 = (
            self.eig_vec_1.copy()
            # .apply_matrix(self.rot_mat, about_point=self.frame_lin_trans.get_center())
            # .shift(self.frame_text.get_center() - self.frame_lin_trans.get_center())
        )
        local_eig_vec_2 = (
            self.eig_vec_2.copy()
            # .apply_matrix(self.rot_mat, about_point=self.frame_lin_trans.get_center())
            # .shift(self.frame_text.get_center() - self.frame_lin_trans.get_center())
        )

        theta = -np.arccos(self.rot_mat[0, 0])
        # self.play(
        #     LaggedStart(
        #         DrawBorderThenFill(plane),
        #         ReplacementTransform(self.eig_vec_1.copy(), local_eig_vec_1),
        #         ReplacementTransform(self.eig_vec_2.copy(), local_eig_vec_2),
        #     )
        #     Rotate(
        #         local_eig_vec_1,
        #         angle=theta,
        #         about_point=self.frame_lin_trans.get_center(),
        #     ),
        #     Rotate(
        #         local_eig_vec_2,
        #         angle=theta,
        #         about_point=self.frame_lin_trans.get_center(),
        #     ),
        #     # local_eig_vec_1.animate.apply_matrix(
        #     #     self.rot_mat, about_point=self.frame_lin_trans.get_center()
        #     # ),
        #     # local_eig_vec_2.animate.apply_matrix(
        #     #     self.rot_mat, about_point=self.frame_lin_trans.get_center()
        #     # ),
        # )
        # )
        self.play(
            local_eig_vec_1.animate.shift(
                self.frame_text.get_center() - self.frame_lin_trans.get_center()
            ),
            local_eig_vec_2.animate.shift(
                self.frame_text.get_center() - self.frame_lin_trans.get_center()
            ),
        )
        self.play(
            Rotate(
                local_eig_vec_1,
                angle=theta,
                about_point=self.frame_text.get_center(),
            ),
            Rotate(
                local_eig_vec_2,
                angle=theta,
                about_point=self.frame_text.get_center(),
            ),
        )
        self.play(Create(plane))

        self.wait(2)

        self.play(
            plane.animate.apply_matrix(lin_trans, about_point=plane_origin_loc),
            rate_func=there_and_back_with_pause,
        )

    def show_lin_trans_and_eigen_vec(self):
        def to_global_coord(point):
            return np.array(point) + self.plane_origin_loc

        plane = self.plane
        lin_trans = self.lin_trans
        eig_vals = self.eig_vals
        eig_vecs = self.eig_vecs

        eig_vec_1 = self.eig_vec_1 = Arrow(
            self.plane_origin_loc,
            np.array([comp for comp in eig_vecs[:, 0]] + [0]) + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
            tip_shape=ArrowTriangleTip,
        )
        eig_vec_1_to_transform = eig_vec_1.copy()
        eig_vec_1_transformed = Arrow(
            self.plane_origin_loc,
            np.array([comp * eig_vals[0] for comp in eig_vecs[:, 0]] + [0])
            + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
            # tip_shape=ArrowTriangleTip,
        )

        eig_vec_2 = self.eig_vec_2 = Arrow(
            self.plane_origin_loc,
            np.array([comp for comp in eig_vecs[:, 1]] + [0]) + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
            tip_shape=ArrowTriangleTip,
        )
        eig_vec_2_to_transform = eig_vec_2.copy()
        eig_vec_2_transformed = Arrow(
            self.plane_origin_loc,
            np.array([comp * eig_vals[1] for comp in eig_vecs[:, 1]] + [0])
            + self.plane_origin_loc,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
            # tip_shape=ArrowTriangleTip,
        )

        v_1 = MathTex(r"\bm v_1", color=ORANGE)
        v_2 = MathTex(r"\bm v_2", color=ORANGE)
        v_1.next_to(eig_vec_1, UP)
        v_2.next_to(eig_vec_2, RIGHT)

        self.play(DrawBorderThenFill(eig_vec_1), DrawBorderThenFill(eig_vec_2))
        self.play(Write(v_1), Write(v_2))

        self.wait(6)

        self.play(
            plane.animate.apply_matrix(lin_trans, about_point=self.plane_origin_loc),
            eig_vec_1_to_transform.animate.apply_matrix(
                lin_trans, about_point=self.plane_origin_loc
            ),
            eig_vec_2_to_transform.animate.apply_matrix(
                lin_trans, about_point=self.plane_origin_loc
            ),
        )

        self.remove(eig_vec_1_to_transform)
        self.add(eig_vec_1_transformed)
        self.add(eig_vec_1)
        self.remove(eig_vec_2_to_transform)
        self.add(eig_vec_2_transformed)
        self.add(eig_vec_2)

        self.wait(2)

        # comp_1_tex = MathTex(r"( \bm x, \bm v_1)", color=BLUE)
        # comp_2_tex = MathTex(r"( \bm x, \bm v_2)", color=BLUE)
        # random_vec_coord = np.array([1, 1, 0])
        # random_vec = Arrow(
        #     self.plane_origin_loc,
        #     np.array(random_vec_coord) + self.plane_origin_loc,
        #     buff=0,
        #     stroke_width=2,
        #     max_tip_length_to_length_ratio=10,
        #     max_stroke_width_to_length_ratio=20,
        #     color=BLUE,
        #     # tip_shape=ArrowTriangleTip,
        # )
        # eig_1 = np.zeros(3)
        # eig_2 = np.zeros(3)
        # eig_1[:2] = eig_vecs[:, 0]
        # eig_2[:2] = eig_vecs[:, 1]
        # comp_1 = eig_1.dot(random_vec_coord)
        # comp_2 = eig_2.dot(random_vec_coord)
        # brace_comp_1 = BraceBetweenPoints(
        #     self.plane_origin_loc,
        #     np.array(comp_1 * eig_1) + self.plane_origin_loc,
        #     buff=0,
        # )
        # line_comp_1 = DashedLine(
        #     to_global_coord(random_vec_coord),
        #     to_global_coord(random_vec_coord - comp_1 * eig_1),
        # )
        # line_comp_1.stroke_width = 0.5
        # line_comp_1.opacity = 0.5
        # line_comp_2 = DashedLine(
        #     to_global_coord(random_vec_coord),
        #     to_global_coord(random_vec_coord - comp_2 * eig_2),
        # )
        # brace_comp_2 = BraceBetweenPoints(
        #     to_global_coord(comp_2 * eig_2), self.plane_origin_loc, buff=0
        # )
        # line_comp_2.stroke_width = 0.5
        # line_comp_2.opacity = 0.5
        # self.play(*[Unwrite(mobject) for mobject in [v_1, v_2]])
        # self.play(
        #     *[
        #         DrawBorderThenFill(mobject)
        #         for mobject in [
        #             random_vec,
        #             line_comp_1,
        #             line_comp_2,
        #             brace_comp_1,
        #             brace_comp_2,
        #         ]
        #     ],
        #     run_time=std_wait_time,
        # )
        # comp_1_tex.next_to(brace_comp_2, LEFT)
        # comp_2_tex.next_to(brace_comp_1, DOWN)
        # self.play(*[Write(mobject) for mobject in [comp_1_tex, comp_2_tex]])
        # comp_1_tex.add_background_rectangle()
        # comp_2_tex.add_background_rectangle()

    def construct(self):

        self.frame_lin_trans = Rectangle(height=6.0, width=6.0)
        self.frame_lin_trans.to_edge(LEFT, buff=1)

        self.frame_text = Rectangle(height=6.0, width=7.0)
        frame_text = self.frame_text
        frame_text.to_edge(RIGHT, buff=1)
        frame_u = Rectangle(height=2, width=14.0)
        large_v_group = VGroup(self.frame_lin_trans, frame_text)
        large_v_group.arrange(RIGHT, buff=1)
        full_group = VGroup(large_v_group)  # frame_u, large_v_group)
        # full_group.arrange(DOWN)
        full_group.center()

        ORIGIN = np.array([0, 0, 0])
        # self.add(full_group)

        # self.add(frame_text)

        title = Tex("Ordinary vectors").scale(1.5)

        all_text_1st_phase = VGroup()

        diff_eq = MathTex(*r"m D^2 f(t) + c D f(t) +kf(t) = F(t)".split(), color=BLUE)
        diff_eq_simple = MathTex(*r"D f(t) = F(t)".split(), color=BLUE)

        problem_vector = MathTex(*r"D \bm{x} = \bm{b}".split(), color=BLUE)
        problem_vector_final = problem_vector.copy()
        problem_vector_replaced_1_a = MathTex(
            *r"\sum_{i=1}^2 ( D \bm{x} , \bm{v}_i ) \bm{v}_i = \sum_{i=1}^2 ( \bm{b} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            *r"\sum_{i=1}^2 ( \bm{x} , D^T \bm{v}_i ) \bm{v}_i = \sum_{i=1}^2 ( \bm{b} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_2_5_a = MathTex(
            *r"\sum_{i=1}^2 ( \bm{x} , D \bm{v}_i ) \bm{v}_i = \sum_{i=1}^2 ( \bm{b} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            *r"\sum_{i=1}^2 ( \bm{x} , \lambda_i \bm{v}_i ) \bm{v}_i = \sum_{i=1}^2 ( \bm{b} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_3_5_a = MathTex(
            *r"\sum_{i=1}^2 \lambda_i ( \bm{x} , \bm{v}_i ) \bm{v}_i = \sum_{i=1}^2 ( \bm{b} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            *r"\lambda_i ( \bm{x} , \bm{v}_i ) = ( \bm{b} , \bm{v}_i ) ,{\quad}i=1,2".split(),
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            *r"( \bm{x} , \bm{v}_i ) = ( \bm{b} , \bm{v}_i ) / \lambda_i ,{\quad}i=1,2".split(),
            color=BLUE,
        )
        easy_coords = problem_vector_replaced_5.copy()
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("{{$D$}} is symmetric {{$(D=D^T)$}}")
        text_lin_trans.get_part_by_tex(r"$(D=D^T)$").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex(
            "Ideally, what is the simplest\\\\effect $D$ could have on some vector $\\bm v$?"
        )
        eigen_problem_vector = MathTex(
            r"{{D \bm v = \lambda \bm v}} {{\to \begin{cases} \bm v_i,\\ \lambda_i, \end{cases} i=1,2}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space = MathTex(
            *r"\bm{x} = \sum_{i=1}^2 ( \bm{x} , \bm{v}_i ) \bm{v}_i".split(),
            color=BLUE
            # *r"\bm{x} = \sum_i( \bm{x} ,\bm{v}_i)\bm{v}_i".split(), color=BLUE
        )
        all_text_1st_phase.add(vec_in_phase_space)

        # self.play(Write(title))
        # self.wait(2)

        self.wait(4)
        # So now we are going to put this analogy to use.
        # Remember, we are trying to understand the Fourier transform relative to differential equations.
        self.play(Write(diff_eq))

        self.wait(8)
        # Let us consider the simplest case.
        self.play(
            ReplacementTransform(VGroup(*diff_eq[5:7], *diff_eq[8:]), diff_eq_simple),
            FadeOut(diff_eq[:5] + diff_eq[7], shift=DOWN),
        )

        operator_symbol = diff_eq_simple.get_parts_by_tex("D")[0]
        lin_operator_label = (
            Tex(r"Linear\\Operator").next_to(operator_symbol, UL).scale(0.8)
        )
        operator_arrow = Arrow(
            lin_operator_label.get_bottom(),
            operator_symbol.get_top(),
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(GrowArrow(operator_arrow), Write(lin_operator_label))

        operator_symbol_in_vec_eq = problem_vector.get_parts_by_tex("D")

        # We will start our exploration with ordinary vectors, meaning the usual arrows that spring to mind.
        # We also switch to more suggestive symbols for this context.
        self.wait(18)
        self.play(
            ReplacementTransform(
                diff_eq_simple,
                problem_vector,
            ),
            FadeOut(operator_arrow),
            FadeOut(lin_operator_label),
        )

        self.play(problem_vector.animate.move_to(frame_text.get_top() + 0.5 * DOWN))
        lin_trans_label = (
            Tex(r"Linear\\Transformation")
            .next_to(operator_symbol_in_vec_eq, UL)
            .scale(0.8)
        )
        example_matrix = (
            MathTex(r"\left[\begin{array}{cc} 3 & 0.3 \\  0.3 & 4\end{array}\right]")
            .move_to(lin_trans_label.get_center())
            .scale(0.8)
        )

        operator_arrow.put_start_and_end_on(
            lin_trans_label.get_bottom() + 0.1 * DOWN,
            operator_symbol_in_vec_eq.get_left(),
        )
        vector_label = (
            Tex("Vectors").next_to(problem_vector[-1], UP, buff=0.7).scale(0.8)
        )
        vector_arrow_1 = operator_arrow.copy().put_start_and_end_on(
            vector_label.get_bottom() + 0.1 * DOWN,
            problem_vector[1].get_top() + 0.1 * UP,
        )
        vector_arrow_2 = operator_arrow.copy().put_start_and_end_on(
            vector_label.get_bottom() + 0.1 * DOWN,
            problem_vector[3].get_top() + 0.1 * UP,
        )

        # Now, \(\bm x\) and \(\bm b\) are vectors.
        # And the linear operator \(D\) is not the differentiation operator but some linear transformation.
        self.play(
            GrowArrow(operator_arrow),
            Write(lin_trans_label),
            GrowArrow(vector_arrow_1),
            GrowArrow(vector_arrow_2),
            Write(vector_label),
        )

        # We can think about it as a matrix.
        # For our purposes, we will assume that \(D\) is symmetric matrix or linear operator.
        # Shortly, we will see why.
        # It is still a linear operator, so the conclusions we will reach regarding
        # operators, in general, will still be valid when we look at the differentiation
        # operator.
        self.wait()
        self.play(ReplacementTransform(lin_trans_label, example_matrix))

        self.wait(4)

        text_lin_trans.next_to(problem_vector, DOWN, buff=1)
        self.play(Write(text_lin_trans))

        self.show_linear_transform()

        self.wait(2)

        self.play(
            *[
                FadeOut(obj)
                for obj in [
                    operator_arrow,
                    example_matrix,
                    vector_arrow_1,
                    vector_arrow_2,
                    vector_label,
                ]
            ]
        )

        question_eigen.next_to(text_lin_trans, DOWN, buff=1)
        eigen_problem_vector.next_to(text_lin_trans, DOWN, buff=1)

        self.wait(7)

        # Looking at this equation, we can think to ourselves, in an ideal world, what would
        # be the slightest effect $D$ could have on a vector.
        self.play(Write(question_eigen))

        self.wait(8)
        # Perhaps one of the least complicated effects it could have is simply multiplying a
        # vector by a constant. Obviously, this can be true for every vector, so we are
        # asking for the vectors that don't change direction after applying the operator
        # $D$.
        # This is the well-known eigenproblem,
        self.play(
            ReplacementTransform(
                question_eigen,
                eigen_problem_vector.get_part_by_tex(r"D \bm v = \lambda \bm v"),
            )
        )
        self.show_lin_trans_and_eigen_vec()

        # and the pairs of vectors and constants found
        # are known as the eigenvectors and eigenvalues of the operator.
        self.wait(4.5)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \begin{cases} \bm v_i,\\ \lambda_i, \end{cases} i=1,2"
                )
            )
        )

        self.wait(2.5)

        # FIXME: Add labels to lambda and v saying eigenvalue and eigenvectors + wait (9 sec)

        # Because we decided that $D$ is symmetric, the eigenvectors will be orthogonal, and
        # the eigenvalues will be real values.

        # FIXME: Add indicate of eigen vectors + wait (11 sec)

        vec_in_phase_space.next_to(eigen_problem_vector, DOWN, buff=1)

        # Now, we can express any vector using the eigenvectors as a basis. This can be done
        # by projecting the vector using the inner product in each direction- Summing the
        # contributions in each direction, we get our vector back.
        self.play(Write(vec_in_phase_space))

        self.wait(1)

        # FIXME: Show projecting components - already exists + wait (8  sec)

        self.play(Indicate(vec_in_phase_space, color=ORANGE))

        # And why is this relevant?
        self.wait(7)

        # Let's see what happens when we use this on the equation.
        # Moving \(D\) to right we must now consider its transpose.
        # Using the fact that $D$ is symmetric, we now have $D v_i$. We know this the same
        # as multiplying $v_i$ by its corresponding eigenvalue. And just like that, we don't
        # have in our problem any $D$s. The problem has become a system of equations.
        # So we can find the components of the unknown vector $\bm x$.

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        self.play(
            ReplacementTransform(problem_vector[0:2], problem_vector_replaced_1_a[2:4]),
            ReplacementTransform(problem_vector[2], problem_vector_replaced_1_a[8]),
            ReplacementTransform(problem_vector[3:], problem_vector_replaced_1_a[11]),
            ReplacementTransform(
                vec_in_phase_space[2:4].copy(), problem_vector_replaced_1_a[0:2]
            ),
            ReplacementTransform(
                vec_in_phase_space[2:4].copy(), problem_vector_replaced_1_a[9:11]
            ),
            ReplacementTransform(
                vec_in_phase_space[5:9].copy(), problem_vector_replaced_1_a[4:8]
            ),
            ReplacementTransform(
                vec_in_phase_space[5:9].copy(), problem_vector_replaced_1_a[12:16]
            ),
        )

        self.wait(1.5)

        problem_vector_replaced_2_a.move_to(problem_vector_replaced_1_a.get_center())
        substitution_pairs = [
            ((0, 2), (0, 2)),
            ((5, None), (5, None)),
            ((2, 3), (4, 5)),
            ((3, 4), (2, 3)),
            ((4, 5), (3, 4)),
        ]
        self.play(
            *[
                ReplacementTransform(
                    problem_vector_replaced_1_a[iin[0] : iin[1]],
                    problem_vector_replaced_2_a[out[0] : out[1]],
                )
                for iin, out in substitution_pairs
            ],
        )

        self.wait(4)

        self.play(
            Indicate(
                text_lin_trans.get_part_by_tex(r"$(D=D^T)$"),
                color=ORANGE,
            ),
        )
        #
        problem_vector_replaced_2_5_a.move_to(problem_vector_replaced_2_a.get_center())
        self.play(
            ReplacementTransform(
                problem_vector_replaced_2_a,
                problem_vector_replaced_2_5_a,
            )
        )

        problem_vector_replaced_3_a.move_to(problem_vector_replaced_2_5_a.get_center())
        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(r"D \bm v = \lambda \bm v"),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            ReplacementTransform(
                problem_vector_replaced_2_5_a,
                problem_vector_replaced_3_a,
            )
        )
        self.wait(2)

        problem_vector_replaced_3_5_a.move_to(problem_vector_replaced_3_a.get_center())
        substitution_pairs = [
            ((0, 1), (0, 1)),
            ((5, None), (5, None)),
            ((4, 5), (1, 2)),
            ((3, 4), (4, 5)),
            ((2, 3), (3, 4)),
            ((1, 2), (2, 3)),
        ]
        self.play(
            *[
                ReplacementTransform(
                    problem_vector_replaced_3_a[iin[0] : iin[1]],
                    problem_vector_replaced_3_5_a[out[0] : out[1]],
                )
                for iin, out in substitution_pairs
            ],
        )

        self.wait(2)

        problem_vector_replaced_4_a.move_to(problem_vector_replaced_3_5_a.get_center())
        substitution_pairs = [
            ((1, 7), (0, 6)),
            ((10, 15), (7, 12)),
            ((8, 9), (6, 7)),
        ]
        self.play(
            *[
                ReplacementTransform(
                    problem_vector_replaced_3_5_a[iin[0] : iin[1]],
                    problem_vector_replaced_4_a[out[0] : out[1]],
                )
                for iin, out in substitution_pairs
            ],
            *[
                FadeOut(problem_vector_replaced_3_5_a[i], shift=DOWN)
                for i in [0, 7, 9, 15]
            ],
            Write(problem_vector_replaced_4_a[12:]),
        )
        self.wait(2)

        problem_vector_replaced_5.move_to(problem_vector_replaced_4_a.get_center())
        substitution_pairs = [
            ((1, 12), (0, 11)),
            ((0, 1), (11, 13)),
            ((12, 13), (13, 14)),
        ]
        self.play(
            *[
                ReplacementTransform(
                    problem_vector_replaced_4_a[iin[0] : iin[1]],
                    problem_vector_replaced_5[out[0] : out[1]],
                )
                for iin, out in substitution_pairs
            ],
        )
        self.wait(2)
        self.play(
            Indicate(
                problem_vector_replaced_5[0:5],
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space[3:8],
                color=ORANGE,
            ),
        )

        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    problem_vector_replaced_5,
                    # vec_in_phase_space,
                    text_lin_trans,
                    eigen_problem_vector,
                ]
            ],
            vec_in_phase_space.animate.shift(0.4 * DOWN),
        )
        self.wait()

        # What is the geometric meaning of this?

        vec_in_original_coords = MathTex(
            *r"\bm{x} = \sum_{i=1}^2 ( \bm{x} , \hat{\bm{e}}_i ) \hat{\bm{e}}_i".split(),
            color=BLUE,
        )
        vec_in_original_coords.move_to(
            vec_in_phase_space.get_center(), coor_mask=[0, 1, 0]
        )
        vec_in_original_coords.move_to(self.plane.get_center(), coor_mask=[1, 0, 0])

        # Using our original description of the problem the vectors are describing using the
        # two unit vectors e_1 and e_2.
        self.play(Write(vec_in_original_coords))

        # If we choose the the eigenvectors of the linear transformation as basis vectors,
        # the linear transformation is much simpler being just a contraction or extension in
        # perpendicular directions.

        self.show_linear_transform_new_coords()

        vector_b_trans = Arrow(
            self.plane_new_coords.get_center(),
            self.plane_new_coords.get_center() + np.append(self.b_trans, 0),
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=YELLOW,
        )
        vector_sol_trans = Arrow(
            self.plane_new_coords.get_center(),
            self.plane_new_coords.get_center() + np.append(self.sol_trans, 0),
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=BLUE,
        )

        self.play(GrowArrow(vector_b_trans))

        easy_coords.next_to(vec_in_phase_space, UP, buff=0.5)
        self.play(Write(easy_coords))
        easy_coords.add_background_rectangle()
        self.play(GrowArrow(vector_sol_trans))

        self.wait()
        self.play(
            self.plane_new_coords.animate.apply_matrix(
                self.diag_lin_trans, about_point=self.plane_new_coords.get_center()
            ),
            vector_sol_trans.animate.apply_matrix(
                self.diag_lin_trans, about_point=self.plane_new_coords.get_center()
            ),
        )

        self.wait()
        self.play(
            self.plane_new_coords.animate.apply_matrix(
                np.linalg.inv(self.diag_lin_trans),
                about_point=self.plane_new_coords.get_center(),
            ),
            vector_sol_trans.animate.apply_matrix(
                np.linalg.inv(self.diag_lin_trans),
                about_point=self.plane_new_coords.get_center(),
            ),
        )

        self.wait()
        self.play(
            self.plane.animate.apply_matrix(
                np.linalg.inv(self.lin_trans), about_point=self.plane.get_center()
            )
        )

        theta = np.arccos(self.rot_mat[0, 0])

        vector_b_orig = vector_b_trans.copy()
        vector_sol_orig = vector_sol_trans.copy()

        self.play(
            vector_b_orig.animate.shift(
                -(self.frame_text.get_center() - self.frame_lin_trans.get_center())
            ),
            vector_sol_orig.animate.shift(
                -(self.frame_text.get_center() - self.frame_lin_trans.get_center())
            ),
        )
        self.play(
            Rotate(
                vector_b_orig,
                angle=theta,
                about_point=self.frame_lin_trans.get_center(),
            ),
            Rotate(
                vector_sol_orig,
                angle=theta,
                about_point=self.frame_lin_trans.get_center(),
            ),
        )

        self.wait()

        problem_vector_final.move_to(vec_in_original_coords, coor_mask=[1, 0, 0])
        problem_vector_final.move_to(easy_coords, coor_mask=[0, 1, 0])
        self.play(Write(problem_vector_final))
        problem_vector_final.add_background_rectangle()

        self.wait()

        self.play(
            self.plane.animate.apply_matrix(
                self.lin_trans,
                about_point=self.frame_lin_trans.get_center(),
            ),
            vector_sol_orig.animate.apply_matrix(
                self.lin_trans,
                about_point=self.frame_lin_trans.get_center(),
            ),
        )

        self.wait(5)


class TestLatex(Scene):
    def construct(self):

        problem_vector = MathTex(r"{{D\bm x}} {{=}} {{\bm b}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            *r"\sum_i ( D \bm{x} , \bm{v}_i )\bm{v}_i=\sum_i(\bm{b},\bm{v}_i)\bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            *r"\sum_i ( \bm{x} , D^T \bm{v}_i )\bm{v}_i=\sum_i(\bm{b},\bm{v}_i)\bm{v}_i".split(),
            color=BLUE,
        )
        problem_vector_replaced_2_5_b = MathTex(
            *r"\sum_i ( \bm{x} , D \bm{v}_i )\bm{v}_i=\sum_i(\bm{b},\bm{v}_i)\bm{v}_i".split(),
            color=BLUE,
        )

        self.add(problem_vector_replaced_1_a)

        self.wait(1)

        self.play(
            TransformMatchingTex(
                problem_vector_replaced_1_a,
                problem_vector_replaced_2_a,
                key_map={"D": "D^T"},
            ),
        )
        self.wait()

        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_a,
                problem_vector_replaced_2_5_b,
                key_map={"D^T": "D"},
            ),
        )


class Move(Scene):
    def __init__(self):
        config.pixel_height = 1080  #
        config.pixel_width = 1080  #
        # config.frame_height = 5
        # config.frame_width = 5
        config.frame_height = 10
        config.frame_width = 10
        Scene.__init__(self)

    def construct(self):

        # dot = Dot(ORIGIN)
        # arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        # numberplane = NumberPlane()
        # origin_text = Text("(0, 0)").next_to(dot, DOWN)
        # tip_text = Text("(2, 2)").next_to(arrow.get_end(), RIGHT)
        # self.add(numberplane, dot, arrow, origin_text, tip_text)

        lin_trans = np.array([[3, 0.3], [0.3, 4]]) / 2.5
        plane = NumberPlane(
            x_range=[-5, 5, 0.5], y_range=[-4, 4, 0.5], x_length=5, y_length=5
        )

        eig_vals, eig_vecs = np.linalg.eig(lin_trans)

        eig_vec_1 = Arrow(
            ORIGIN,
            [comp for comp in eig_vecs[:, 0]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
        )
        eig_vec_1_to_transform = eig_vec_1.copy()
        eig_vec_1_transformed = Arrow(
            ORIGIN,
            [comp * eig_vals[0] for comp in eig_vecs[:, 0]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
        )

        eig_vec_2 = Arrow(
            ORIGIN,
            [comp for comp in eig_vecs[:, 1]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
        )
        eig_vec_2_to_transform = eig_vec_2.copy()
        eig_vec_2_transformed = Arrow(
            ORIGIN,
            [comp * eig_vals[1] for comp in eig_vecs[:, 1]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
        )
        v_1 = MathTex(r"\bm v_1", color=ORANGE)
        v_2 = MathTex(r"\bm v_2", color=ORANGE)
        v_1.next_to(eig_vec_1, UP)
        v_2.next_to(eig_vec_2, RIGHT)

        self.play(DrawBorderThenFill(plane), run_time=2)
        self.wait(2)
        self.play(DrawBorderThenFill(eig_vec_1), DrawBorderThenFill(eig_vec_2))
        self.play(Write(v_1), Write(v_2))

        self.play(
            plane.animate.apply_matrix(lin_trans),
            eig_vec_1_to_transform.animate.apply_matrix(lin_trans),
            eig_vec_2_to_transform.animate.apply_matrix(lin_trans),
        )
        self.remove(eig_vec_1_to_transform)
        self.add(eig_vec_1_transformed)
        self.add(eig_vec_1)
        self.remove(eig_vec_2_to_transform)
        self.add(eig_vec_2_transformed)
        self.add(eig_vec_2)

        self.wait(2)

        comp_1_tex = MathTex(r"( \bm x, \bm v_1)", color=BLUE)
        comp_2_tex = MathTex(r"( \bm x, \bm v_2)", color=BLUE)
        random_vec_coord = np.array([1, 1, 0])
        random_vec = Arrow(
            ORIGIN,
            random_vec_coord,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=BLUE,
        )
        eig_1 = np.zeros(3)
        eig_2 = np.zeros(3)
        eig_1[:2] = eig_vecs[:, 0]
        eig_2[:2] = eig_vecs[:, 1]
        comp_1 = eig_1.dot(random_vec_coord)
        comp_2 = eig_2.dot(random_vec_coord)
        brace_comp_1 = BraceBetweenPoints(ORIGIN, comp_1 * eig_1, buff=0)
        line_comp_1 = DashedLine(random_vec_coord, random_vec_coord - comp_1 * eig_1)
        line_comp_1.stroke_width = 0.5
        line_comp_1.opacity = 0.5
        line_comp_2 = DashedLine(random_vec_coord, random_vec_coord - comp_2 * eig_2)
        brace_comp_2 = BraceBetweenPoints(comp_2 * eig_2, ORIGIN, buff=0)
        line_comp_2.stroke_width = 0.5
        line_comp_2.opacity = 0.5
        self.play(*[Unwrite(mobject) for mobject in [v_1, v_2]])
        self.play(
            *[
                DrawBorderThenFill(mobject)
                for mobject in [
                    random_vec,
                    line_comp_1,
                    line_comp_2,
                    brace_comp_1,
                    brace_comp_2,
                ]
            ],
            run_time=std_wait_time,
        )
        comp_1_tex.next_to(brace_comp_2, LEFT)
        comp_2_tex.next_to(brace_comp_1, DOWN)
        self.play(*[Write(mobject) for mobject in [comp_1_tex, comp_2_tex]])
        comp_1_tex.add_background_rectangle()
        comp_2_tex.add_background_rectangle()

        self.wait(10)

        # f_tex = r"f(x)"
        # equation = MathTex("dA", "\\approx", f_tex, "dx")
        # # equation.to_edge(RIGHT).shift(3*UP)
        # deriv_equation = MathTex("{dA", "\\over \\,", "dx}", "\\approx", f_tex)
        # deriv_equation.move_to(equation, UP + LEFT)
        #
        # self.play(
        #     *[
        #         ReplacementTransform(
        #             equation.get_part_by_tex(tex),
        #             deriv_equation.get_part_by_tex(tex),
        #             run_time=10,
        #         )
        #         for tex in ("dA", "approx", f_tex, "dx")
        #     ]
        #     + [Write(deriv_equation.get_part_by_tex("over"))]
        # )


class FunctionsAsVectors(Scene):
    def construct(self):

        axes = Axes(
            x_range=[-2, 6, 1], x_length=7, y_range=[-5, 5, 1], y_length=5, tips=False
        )
        frame_text = Rectangle(height=6, width=7.0)
        frame_ur = Rectangle(height=2, width=7.0)
        frame_ul = Rectangle(height=2, width=7.0)
        up_frames = VGroup(frame_ul, frame_ur)
        up_frames.arrange(RIGHT, buff=0.7)
        frame_text.to_edge(RIGHT, buff=0.7)
        large_v_group = VGroup(axes, frame_text)
        large_v_group.arrange(RIGHT, buff=1)
        full_group = VGroup(up_frames, large_v_group)
        full_group.arrange(DOWN)
        full_group.center()
        # self.add(full_group)
        # self.add(frame_text)

        title = Tex(r"Vector/Function analogy").scale(1.5)
        title.move_to(frame_ur.get_center())

        axes_graph = axes.get_graph(
            lambda x: 4 * np.sin(np.pi / 3.5 * (x)), x_range=[-2, 6], color=ORANGE
        )
        vec_vals = [ValueTracker(0) for i in range(4)]

        changing_vec_tex = always_redraw(
            lambda: MathTex(
                "\\bm u = \\left\\{{\\begin{{array}}{{c}} {0:.2f} \\\\ {1:.2f} \\\\ {2:.2f} \\\\ {3:.2f} \\end{{array}}\\right\\}}".format(
                    *[val_tracker.get_value() for val_tracker in vec_vals]
                )
            )
            .move_to(frame_text.get_center())
            .set_color(BLUE)
        )
        vec_func_comp = MathTex(
            r"\bm u\colon \{1,2,3,4\}\to \mathbb R", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_comp_complex = MathTex(
            r"\bm u\colon \{1,2,3,4\}\to \mathbb C", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_int = MathTex(
            r"\bm u\colon \mathbb Z\to \mathbb R", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_real = MathTex(r"f\colon \mathbb R\to \mathbb R", color=BLUE).move_to(
            frame_text.get_center()
        )
        vec_func_complex = MathTex(
            r"f\colon \mathbb R\to \mathbb C", color=BLUE
        ).move_to(frame_text.get_center())
        inner_vec = MathTex(r"{{(\bm u, \bm v)=\sum_i u_i}} v_i", color=BLUE)
        inner_vec_comp = MathTex(
            r"{{(\bm u, \bm v)=\sum_i u_i}} \overline{ v_i }\,", color=BLUE
        )
        inner_func = MathTex(
            r"{{(f, g) = \int_I f(x)}}g(x){{\,\mathrm d x}}", color=BLUE
        )
        inner_func_comp = MathTex(
            r"{{(f, g) = \int_I f(x)}}\overline{g(x)}{{\,\mathrm d x }}\,",
            color=BLUE,
        )

        # Positioning
        # ==================================================================================
        all_phrases = VGroup(vec_func_comp, inner_vec, vec_func_int, inner_func)
        all_phrases.arrange(DOWN, buff=1)
        all_phrases.move_to(frame_text.get_center())

        vec_func_real.move_to(vec_func_int.get_center())
        vec_func_comp_complex.move_to(vec_func_comp.get_center())
        vec_func_complex.move_to(vec_func_real.get_center())
        inner_vec_comp.move_to(inner_vec.get_center())
        inner_func_comp.move_to(inner_func.get_center())

        y_0 = axes.coords_to_point(0, 0)[1]
        dots_vec = [
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(1, vec_vals[0].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(2, vec_vals[1].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(3, vec_vals[2].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(4, vec_vals[3].get_value()), color=ORANGE
                )
            ),
        ]
        dots_int_func = [
            Dot(
                point=axes.coords_to_point(i, 4 * np.sin(np.pi / 3.5 * (i))),
                color=ORANGE,
            )
            for i in [-2, -1, 0, 5, 6]
        ]
        lines_int_func = [
            DashedLine(
                axes.coords_to_point(i, 0),
                axes.coords_to_point(i, 4 * np.sin(np.pi / 3.5 * (i))),
            )
            for i in [-2, -1, 0, 5, 6]
        ]
        lines_vec = [
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(1, 0),
                    axes.coords_to_point(1, vec_vals[0].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(2, 0),
                    axes.coords_to_point(2, vec_vals[1].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(3, 0),
                    axes.coords_to_point(3, vec_vals[2].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(4, 0),
                    axes.coords_to_point(4, vec_vals[3].get_value()),
                )
            ),
        ]

        # dots_vec = [
        #     always_redraw(
        #         lambda: Dot(point=axes.coords_to_point(1, vec_vals[0])),
        #     ),
        #     always_redraw(
        #         lambda: Dot(point=axes.coords_to_point(2, vec_vals[1])),
        #     ),
        # ]

        self.play(Write(title))
        self.wait(1)
        self.play(Create(axes))
        self.play(Write(changing_vec_tex))
        self.play(
            *[DrawBorderThenFill(line) for line in lines_vec],
            *[DrawBorderThenFill(dot) for dot in dots_vec],
        )
        np.random.seed(42)
        # self.play(
        #     *[
        #         value_tracker.animate.set_value(8 * np.random.random() - 4)
        #         for i, value_tracker in enumerate(vec_vals)
        #     ],
        #     rate_func=linear,
        #     run_time=4,
        # )
        self.play(
            *[
                value_tracker.animate.set_value(8 * np.random.random() - 4)
                for i, value_tracker in enumerate(vec_vals)
            ],
            rate_func=linear,
            run_time=4,
        )
        self.play(
            *[
                value_tracker.animate.set_value(4 * np.sin(np.pi / 3.5 * (i + 1)))
                for i, value_tracker in enumerate(vec_vals)
            ],
            rate_func=linear,
            run_time=4,
        )
        self.play(ReplacementTransform(changing_vec_tex, vec_func_comp))
        self.wait(2)
        self.play(Write(inner_vec))
        self.wait(1)
        self.play(
            *[DrawBorderThenFill(line) for line in lines_int_func],
            *[DrawBorderThenFill(dot) for dot in dots_int_func],
        )
        self.play(TransformFromCopy(vec_func_comp, vec_func_int))
        self.wait(1)
        self.play(Create(axes_graph))
        self.play(ReplacementTransform(vec_func_int, vec_func_real))
        self.wait(1)
        self.play(TransformFromCopy(inner_vec, inner_func))
        self.wait(1)
        self.play(
            ReplacementTransform(vec_func_comp, vec_func_comp_complex),
            ReplacementTransform(vec_func_real, vec_func_complex),
        )
        self.wait(1)
        self.play(
            ReplacementTransform(inner_vec, inner_vec_comp),
            ReplacementTransform(inner_func, inner_func_comp),
        )
        self.wait(5)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    inner_vec_comp,
                    inner_func_comp,
                    title,
                    vec_func_comp_complex,
                    vec_func_complex,
                ]
            ],
            Uncreate(axes),
            Uncreate(axes_graph),
            *[
                Uncreate(mobject)
                for mobject in dots_vec + dots_int_func + lines_vec + lines_int_func
            ],
        )


class FourierSeriesScene(Scene):
    def construct(self):

        big_frame = Rectangle(height=6.0, width=14.0)
        frame_u = Rectangle(height=2, width=14.0)
        full_group = VGroup(frame_u, big_frame)
        full_group.arrange(DOWN)
        full_group.center()

        title = Tex("Periodic functions").scale(1.5)
        title.to_edge(UP, buff=1)
        self.add(title)
        #
        all_text_1st_phase = VGroup()
        #
        problem_vector = MathTex(r"{{Df}} {{=}} {{g}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} }}( {{Df}}, \varphi_n) \varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} ( {{g}}, \varphi_n) \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} (}} {{D}}{{f}},{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} (}} {{f}}, {{D}}^\dagger{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, }}{D}^\dagger{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, }}-{{D}}{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, -}}D{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, -}}\frac{2\pi in}{L}{{\varphi_n)}}{{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_b = MathTex(
            r"\sum_{n=-\infty}^{+\infty} {{( f,}} {{-\frac{2\pi in}{L} }} {{\varphi_n)}}\varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} {{( g, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_3_5_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{-\frac{2\pi inx}{L} }}{{( f,}} {{\varphi_n)}}{{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_5_b = MathTex(
            r"\sum_{n=-\infty}^{+\infty} {{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}}\varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} {{( g, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}} {{=}} {{( g, \varphi_n)}},\quad n\in\mathbb Z",
            color=BLUE,
        )
        problem_vector_replaced_4_b = MathTex(
            r"{{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}} {{=}} {{( g, \varphi_n)}}{{,\quad n\in\mathbb Z}}",
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            r"{{( f, \varphi_n)}} {{=}} {{i}}{{( g, \varphi_n)}}/{{\frac{L}{2\pi n x} }}{{,\quad n\in\mathbb Z}}",
            color=BLUE,
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("$D$ is anti-Hermitian {{($D^\dagger = -D$)}}")
        text_lin_trans.get_part_by_tex(r"($D^\dagger = -D$)").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{-iD \varphi(x) = \lambda \varphi(x)}} {{\to \varphi_n(x) = \frac{e^\frac{2\pi inx}{L} }{\sqrt{2\pi} },\quad n \in \mathbb Z}}",
            color=BLUE,
        )
        eigen_problem_vector_2 = MathTex(
            r"D \varphi(x) = i\lambda \varphi(x)",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space_a = MathTex(
            r"{{f(x) = \sum_{n=-\infty}^{+\infty} }}{{\int_{-L/2}^{L/2} \frac{e^\frac{-2\pi i n y}{L} }{\sqrt{2\pi} }\mathrm d y}} \frac{e^\frac{2\pi i n x}{L} }{\sqrt{2\pi} }",
            color=BLUE,
        )
        vec_in_phase_space = MathTex(
            r"{{f(x) = \sum_{n=-\infty}^{+\infty} }} {{(f, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        all_text_1st_phase.add(vec_in_phase_space)

        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(big_frame.get_center())
        braces = Brace(vec_in_phase_space, RIGHT)
        eq_text = braces.get_text("Fourier series")

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())

        eigen_problem_vector_2.move_to(
            eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = \lambda \varphi(x)")
        )

        vec_in_phase_space_a.move_to(vec_in_phase_space)

        self.play(Write(title))
        self.wait(2)
        self.add(problem_vector)
        self.wait(2)
        self.play(Write(text_lin_trans))
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"-iD \varphi(x) = \lambda \varphi(x)"
                )
            )
        )
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \varphi_n(x) = \frac{e^\frac{2\pi inx}{L} }{\sqrt{2\pi} },\quad n \in \mathbb Z"
                )
            )
        )
        self.wait(2)
        self.play(Write(vec_in_phase_space_a))
        self.wait(2)
        self.play(ReplacementTransform(vec_in_phase_space_a, vec_in_phase_space))

        self.wait(2)
        self.play(GrowFromCenter(braces), Write(eq_text))
        self.wait(1)
        self.play(Indicate(vec_in_phase_space, color=ORANGE))
        self.wait(2)
        self.play(
            # ReplacementTransform(problem_vector.get_part_by_tex(r"f"), test),
            TransformMatchingTex(problem_vector, problem_vector_replaced_1_a),
        )
        self.wait(2)
        self.remove(problem_vector_replaced_1_a)
        self.add(problem_vector_replaced_1_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_1_b,
                problem_vector_replaced_2_a,
                run_time=std_wait_time,
            )
        )
        self.remove(problem_vector_replaced_2_a)
        self.add(problem_vector_replaced_2_b)
        self.wait(2)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_b,
                problem_vector_replaced_2_5_a,
                run_time=std_wait_time,
            )
        )
        self.remove(problem_vector_replaced_2_5_a)
        self.add(problem_vector_replaced_2_5_b)
        self.wait(2)

        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(
                    r"-iD \varphi(x) = \lambda \varphi(x)"
                ),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_5_b,
                problem_vector_replaced_3_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_a)
        self.add(problem_vector_replaced_3_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_b,
                problem_vector_replaced_3_5_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_5_a)
        self.add(problem_vector_replaced_3_5_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_5_b,
                problem_vector_replaced_4_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_4_a)
        self.add(problem_vector_replaced_4_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_4_b,
                problem_vector_replaced_5,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.play(
            Indicate(
                problem_vector_replaced_5.get_part_by_tex(r"( f, \varphi_n)"),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space.get_part_by_tex(r"(f, \varphi_n)"),
                color=ORANGE,
            ),
        )

        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    title,
                    problem_vector_replaced_5,
                    text_lin_trans,
                    eigen_problem_vector,
                    vec_in_phase_space,
                    braces,
                    eq_text,
                ]
            ]
        )

        self.wait(10)


class FourierIntegralScene(Scene):
    def construct(self):

        big_frame = Rectangle(height=6.0, width=14.0)
        frame_u = Rectangle(height=2, width=14.0)
        full_group = VGroup(frame_u, big_frame)
        full_group.arrange(DOWN)
        full_group.center()

        # self.add(big_frame)
        # self.add(frames)

        title = Tex("Functions").scale(1.5)
        title.move_to(frame_u.get_center())
        #
        all_text_1st_phase = VGroup()
        #
        problem_vector = MathTex(r"{{D}} {{f}} {{=}} {{g}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            r"\int_{-\infty}^{+\infty} ({{D}}{{f}}, \varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w {{=}} \int_{-\infty}^{+\infty} ({{g}}, \varphi) \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{\int_{-\infty}^{+\infty} }} {{({{D}}f,\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\int_{-\infty}^{+\infty} }} {{(f,}}-{{D}}{{\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\int_{-\infty}^{+\infty}(f,-D\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\int_{-\infty}^{+\infty}(f,-iw\varphi)}}{{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_3_b = MathTex(
            r"\int_{-\infty}^{+\infty} {{(f,-iw\varphi)}}\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w {{=}} \int_{-\infty}^{+\infty} {{(g,\varphi)}} \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{-iw(f,\varphi)}} {{=}} {{(g,\varphi)}},\quad w\in \mathbb R",
            color=BLUE,
        )
        problem_vector_replaced_4_b = MathTex(
            r"-{{i}}{{w}}{{(f,\varphi)}} {{=}} {{(g,\varphi)}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            r"{{(f,\varphi)}} {{=}} {{i}}{{(g,\varphi)}}/{{w}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        problem_vector_replaced_6 = MathTex(
            r"{{[\mathcal F(f)](w)}} {{=}} {{i}}{{[\mathcal F(g)](w)}}/{{w}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("$D$ is anti-Hermitian {{($D^\dagger = -D$)}}")
        text_lin_trans.get_part_by_tex(r"($D^\dagger = -D$)").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{-iD \varphi(x) = w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        eigen_problem_vector_1_b = MathTex(
            r"-{{i}}{{D \varphi(x) =}} {{w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)
        eigen_problem_vector_2_a = MathTex(
            r"{{D \varphi(x) =}} i{{w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        eigen_problem_vector_2_b = MathTex(
            r"{{D \varphi(x) = iw \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space = MathTex(
            r"f(x) = \int_{-\infty}^{+\infty} {{[\mathcal F(f)](w)}} \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        all_text_1st_phase.add(vec_in_phase_space)
        vec_in_phase_space_2 = MathTex(
            r"f(x) = \int_{-\infty}^{+\infty} {{(f, \varphi)}} \varphi(w)\mathrm d w",
            color=BLUE,
        )

        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(big_frame.get_center())

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())
        problem_vector_replaced_6.move_to(problem_vector.get_center())

        vec_in_phase_space_2.move_to(vec_in_phase_space.get_center())

        eigen_problem_vector_1_b.move_to(eigen_problem_vector)
        eigen_problem_vector_2_a.move_to(eigen_problem_vector)
        eigen_problem_vector_2_b.move_to(eigen_problem_vector)

        braces = Brace(vec_in_phase_space_2, RIGHT)
        eq_text = braces.get_text("Fourier integral")

        self.play(Write(title))
        self.wait(2)
        self.add(problem_vector)
        self.wait(2)
        self.play(Write(text_lin_trans))
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = w \varphi(x)")
            )
        )
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R"
                )
            )
        )
        self.wait(2)
        self.play(Write(vec_in_phase_space))
        self.wait(2)
        self.play(
            ReplacementTransform(vec_in_phase_space, vec_in_phase_space_2), run_time=2
        )
        self.remove(vec_in_phase_space)
        self.add(vec_in_phase_space_2)

        self.play(GrowFromCenter(braces), Write(eq_text))

        self.wait(2)
        self.play(Indicate(vec_in_phase_space_2, color=ORANGE))
        self.wait(1)
        self.play(
            # ReplacementTransform(problem_vector.get_part_by_tex(r"f"), test),
            TransformMatchingTex(problem_vector, problem_vector_replaced_1_a),
        )
        self.wait(2)
        self.remove(problem_vector_replaced_1_a)
        self.add(problem_vector_replaced_1_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_1_b,
                problem_vector_replaced_2_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_2_a)
        self.add(problem_vector_replaced_2_b)

        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = w \varphi(x)"),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_b,
                problem_vector_replaced_3_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_a)
        self.add(problem_vector_replaced_3_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_b,
                problem_vector_replaced_4_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_4_a)
        self.add(problem_vector_replaced_4_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_4_b,
                problem_vector_replaced_5,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.play(
            ReplacementTransform(problem_vector_replaced_5, problem_vector_replaced_6)
        )

        self.wait(2)
        self.play(
            Indicate(
                problem_vector_replaced_6.get_part_by_tex(r"[\mathcal F(f)](w)"),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space_2.get_part_by_tex(r"(f, \varphi)"),
                color=ORANGE,
            ),
        )
        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    title,
                    problem_vector_replaced_6,
                    text_lin_trans,
                    eigen_problem_vector,
                    vec_in_phase_space_2,
                    braces,
                    eq_text,
                ]
            ]
        )

        self.wait(10)


class SplitScreenDerivativeMultiplication(Scene):
    def construct(self):

        # Function graph setup
        # ==================================================================================
        axes_func = Axes(
            x_range=[-6, 6],
            y_range=[-2, 2],
            x_length=5,
            y_length=4,
            tips=False,
            x_axis_config={"include_ticks": False},
            y_axis_config={"include_ticks": False},
        )
        box_axes = SurroundingRectangle(axes_func, buff=0)
        magnitude = ValueTracker(1.0)
        theta = ValueTracker(0.0)
        omega = 3

        sin_func_og = axes_func.get_graph(
            lambda t: np.real(np.exp(1j * omega * t)), stroke_opacity=0.5
        )
        sin_func = always_redraw(
            lambda: axes_func.get_graph(
                lambda t: np.real(
                    magnitude.get_value()
                    * np.exp(1j * theta.get_value())
                    * np.exp(1j * omega * t)
                ),
                x_range=[-6, 6],
            )
        )
        full_plot = VGroup(sin_func, box_axes, axes_func, sin_func_og)

        # Complex plane setup
        # ==================================================================================

        plane = ComplexPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=5,
            y_length=4,
        )
        # plane_polar = ComplexPlane(
        #     x_range=[-2, 2],
        #     y_range=[-2, 2],
        #     x_length=5,
        #     y_length=4,
        #     axis_config={"include_ticks": False, "numbers_to_exclude": [-2, 2]},
        # )

        box_plane = SurroundingRectangle(plane, buff=0)

        full_plane = VGroup(plane, box_plane)  # , labels)

        everything = VGroup(full_plot, full_plane)
        everything.arrange(RIGHT, buff=1)
        everything.center()

        labels = plane.get_axis_labels(x_label="\mathrm{Re}", y_label="\mathrm{Im}")
        labels[1].shift(0.5 * DOWN)
        labels[0].shift(0.7 * LEFT)

        self.add(full_plane, labels)
        d1 = always_redraw(
            lambda: Dot(
                plane.n2p(magnitude.get_value() * np.exp(1j * theta.get_value())),
                color=YELLOW,
            )
        )
        self.add(
            d1,
        )
        np.random.seed(40)
        magnitude_random_values = 1.5 * np.random.rand(4)
        theta_random_values = 2 * np.pi * np.random.rand(4)

        # Animation
        # ==================================================================================

        self.add(sin_func, box_axes, sin_func_og)
        self.play(
            magnitude.animate.set_value(1.4), run_time=2, rate_func=there_and_back
        )
        self.wait(1)
        self.play(theta.animate.set_value(5), run_time=4, rate_func=there_and_back)
        self.wait(1)
        for i in range(4):
            self.play(
                magnitude.animate.set_value(magnitude_random_values[i]),
                theta.animate.set_value(theta_random_values[i]),
                run_time=3,
                rate_func=smooth,
            )
        self.play(
            magnitude.animate.set_value(1.2),
            theta.animate.set_value(np.pi / 3),
            run_time=3,
            rate_func=smooth,
        )
        self.wait()


class FourierSeriesExplainer(ThreeDScene):
    def setup_this_scene(self):

        self.rotated_camera_position = [2, 1.5, 1]
        self.rotated_camera_direction = self.rotated_camera_position / np.linalg.norm(
            self.rotated_camera_position
        )

    def func_to_model(self, x, period=1):

        k = np.round(x / period)
        x = x - period * k

        # amplitude = 1
        # val_saw = -amplitude + (x + period / 2) ** 2 * (2 * amplitude) / period
        # val_square = amplitude / 2 * x ** 3 if -period / 4 <= x <= period / 4 else 0

        amplitude = 1.5
        val = amplitude * np.sin(2 * np.pi * x / period) if 0 <= x <= period / 2 else 0

        return val

    def coeffs_func_to_model(self, n_coeffs):

        amplitude = 1.5
        a_coeffs = [
            -2 * amplitude / np.pi * 1 / (n ** 2 - 1) if n % 2 == 0 else 0
            for n in range(n_coeffs + 1)
        ]
        b_coeffs = [0 if n != 1 else amplitude / 2 for n in range(n_coeffs + 1)]

        c_coeffs = [
            0.5 * (a_coeffs[abs(i)] - 1j * b_coeffs[abs(i)])
            if i >= 0
            else 0.5 * (a_coeffs[abs(i)] + 1j * b_coeffs[abs(i)])
            for i in range(-n_coeffs, n_coeffs + 1)
        ]
        return c_coeffs

    def construct(self):

        self.setup_this_scene()
        delta_z = 2.5
        n_points = 500

        axes_func = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-2.5 * 4, 2.5 * 4, 1],
            # z_range=[-1e-4, 1e-4, 1],
            x_length=8,
            y_length=5,
            z_length=2.5 * 6,
            axis_config={"include_ticks": False},
            tips=False,
        )
        axes_coeffs = ThreeDAxes(
            x_range=[-0.5, 0.5, 0.1],
            y_range=[-0.5, 0.5, 0.1],
            z_range=[-2.5 * 4, 2.5 * 4, 2.5],
            x_length=4,
            y_length=4,
            z_length=2.5 * 6,
            # z_axis_config={
            #     "numbers_to_include": list(range(-3, 4)),
            #     "label_direction": RIGHT,
            # },
            # x_axis_config={
            #     "numbers_to_include": [0.5],
            #     "label_direction": np.array([0.45]),
            # },
            # y_axis_config={
            #     "numbers_to_include": [0.5],
            #     "label_direction": np.array([0.45]),
            # },
            tips=False,
        )

        graph_x_values = np.linspace(-1.5, 1.5, 100)
        graph_y_values = [self.func_to_model(x) for x in graph_x_values]
        graph_z_values = [0 for _ in graph_x_values]

        graph = axes_func.get_line_graph(
            x_values=graph_x_values,
            y_values=graph_y_values,
            z_values=graph_z_values,
            add_vertex_dots=False,
        ).set_stroke(color=RED)

        # rects = axes_func.get_riemann_rectangles(
        #     graph=graph, x_range=[-2, 2], dx=0.1, stroke_color=WHITE
        # )
        #
        # graph2 = axes_func.get_parametric_curve(
        #     lambda t: np.array([np.cos(t), np.sin(t), t]),
        #     t_range=[-2 * PI, 2 * PI],
        #     color=RED,
        # )
        all_func = VGroup(axes_func, graph)
        all_coeff = VGroup(axes_coeffs)
        both_graphs = VGroup(all_func, all_coeff)
        both_graphs.arrange(RIGHT)
        both_graphs.center()

        x_label = axes_coeffs.get_x_axis_label(r"\mathrm{Re}")
        x_label.move_to(axes_coeffs.coords_to_point(0.5, 0.15, 0))

        y_label = axes_coeffs.get_y_axis_label(r"\mathrm{Im}")
        y_label.move_to(axes_coeffs.coords_to_point(0.15, 0.5, 0))

        z_label = MathTex("\omega")
        z_label.move_to(axes_coeffs.coords_to_point(0.15, 0.05, 5.5))
        z_label_graph = MathTex("\omega")
        z_label_graph.move_to(axes_func.coords_to_point(0.15, -0.05, 9))

        z_tick_labels_base = []
        for ind in range(-3, 4):
            if ind == 0:
                continue

            z_tick_labels_base.append(MathTex("{0:d}".format(ind), font_size=8.0))
            z_tick_labels_base[-1].move_to(
                axes_coeffs.coords_to_point(-0.05, 0.05, delta_z * ind)
            )
            z_tick_labels_base[-1].scale(0.6)

        def redraw_facing_camera(object):
            def redraw_func():
                new_obj = object.copy()
                new_obj.apply_matrix(
                    self.camera.get_rotation_matrix().T,
                    about_point=new_obj.get_center(),
                )
                return new_obj

            return redraw_func

        x_label_axes_coeffs = always_redraw(redraw_facing_camera(x_label))
        y_label_axes_coeffs = always_redraw(redraw_facing_camera(y_label))
        z_label_axes_coeffs = always_redraw(redraw_facing_camera(z_label))
        z_label_axes_func = always_redraw(redraw_facing_camera(z_label_graph))
        z_tick_labels_all = [
            always_redraw(redraw_facing_camera(tick)) for tick in z_tick_labels_base
        ]

        all_coeff.add(x_label_axes_coeffs, y_label_axes_coeffs)
        # self.add(all_func, all_coeff)

        self.play(
            LaggedStart(
                *[
                    Create(obj)
                    for obj in [
                        axes_func,
                        graph,
                        axes_coeffs,
                        x_label_axes_coeffs,
                        y_label_axes_coeffs,
                    ]
                ],
                run_time=4,
                lag_ratio=0.75,
            )
        )

        x_reference_lines = []
        t = np.linspace(-2, 5, 10)
        for freq in range(-3, 4):
            x_values = t
            y_values = 10 * [0]
            z_values = 10 * [delta_z * freq]

            x_reference_lines.append(
                Line(
                    axes_func.coords_to_point(*[x_values[0], y_values[0], z_values[0]]),
                    axes_func.coords_to_point(
                        *[x_values[-1], y_values[-1], z_values[-1]]
                    ),
                )
            )
            if freq > 0:
                x_reference_lines[-1].set_stroke(color=BLUE, opacity=0.5, width=1.5)
            else:
                x_reference_lines[-1].set_stroke(color=BLUE, opacity=0.8, width=1.5)

        # The camera is auto set to PHI = 0 and THETA = -90
        coeff_dots = []
        coeff_lines = []
        magnitude_trackers = []
        theta_trackers = []
        z_trackers = []

        n_fourier_coeffs = 3
        fourier_coeffs = self.coeffs_func_to_model(n_fourier_coeffs)
        coeff_magnitudes = [np.abs(coeff) for coeff in fourier_coeffs]
        coeff_thetas = [np.angle(coeff) for coeff in fourier_coeffs]

        for i_freq, freq in enumerate(range(-3, 4)):

            def create_dot_line_generator(freq):
                z_tracker = ValueTracker(0)
                magnitude_tracker = ValueTracker(coeff_magnitudes[i_freq])
                theta_tracker = ValueTracker(coeff_thetas[i_freq])

                def dot_generator():
                    return Dot3D(
                        point=axes_coeffs.coords_to_point(
                            np.real(
                                magnitude_tracker.get_value()
                                * np.exp(1j * theta_tracker.get_value())
                            ),
                            np.imag(
                                magnitude_tracker.get_value()
                                * np.exp(1j * theta_tracker.get_value())
                            ),
                            z_tracker.get_value(),
                        ),
                        color=DARK_BLUE,
                    )

                def line_generator():
                    return DashedLine(
                        start=axes_coeffs.coords_to_point(
                            0,
                            0,
                            z_tracker.get_value(),
                        ),
                        end=axes_coeffs.coords_to_point(
                            np.real(
                                magnitude_tracker.get_value()
                                * np.exp(1j * theta_tracker.get_value())
                            ),
                            np.imag(
                                magnitude_tracker.get_value()
                                * np.exp(1j * theta_tracker.get_value())
                            ),
                            z_tracker.get_value(),
                        ),
                    )

                return (
                    magnitude_tracker,
                    theta_tracker,
                    z_tracker,
                    dot_generator,
                    line_generator,
                )

            (
                current_magnitude_tracker,
                current_theta_tracker,
                current_z_tracker,
                dot_generator,
                line_generator,
            ) = create_dot_line_generator(freq)

            current_dot = always_redraw(dot_generator)
            current_line = always_redraw(line_generator)

            magnitude_trackers.append(current_magnitude_tracker)
            theta_trackers.append(current_theta_tracker)
            z_trackers.append(current_z_tracker)

            coeff_dots.append(current_dot)
            coeff_lines.append(current_line)

            # coeff_dots[-1].set_stroke(color=DARK_BLUE, width=0.3)
            # coeff_dots[-1].set_stroke(color=DARK_BLUE)

        # print([coeff_dot.get_center()[2] for coeff_dot in coeff_dots])

        opacity_trackers = [ValueTracker(1) for _ in range(-3, 4)]
        basis_func = []
        t = np.linspace(-1.5, 1.5, n_points)
        for freq in range(-3, 4):

            def create_func_generator(freq, axes):
                def func_generator():
                    x_values = t
                    y_values = np.real(
                        magnitude_trackers[freq + 3].get_value()
                        * np.exp(
                            1j * theta_trackers[freq + 3].get_value()
                            + 1j * freq * t * 2 * np.pi / 1
                        )
                    )
                    z_values = n_points * [z_trackers[freq + 3].get_value()]
                    func = axes.get_line_graph(
                        x_values=x_values,
                        y_values=y_values,
                        z_values=z_values,
                        line_color=BLUE,
                        add_vertex_dots=False,
                    )
                    func.set_stroke(
                        color=DARK_BLUE,
                        opacity=opacity_trackers[freq + 3].get_value(),
                    )
                    # if freq > 0:
                    #     func.set_stroke(
                    #         color=DARK_BLUE,
                    #         opacity=opacity_trackers[freq + 3].get_value(),
                    #         width=0.5,
                    #     )
                    # else:
                    #     func.set_stroke(
                    #         color=DARK_BLUE,
                    #         opacity=opacity_trackers[freq + 3].get_value(),
                    #     )
                    return func

                return func_generator

            basis_func.append(
                always_redraw(
                    create_func_generator(
                        freq,
                        axes_func,  # , magnitude_trackers, theta_trackers, z_trackers
                    )
                )
            )

        self.wait()
        # self.add(*basis_func)
        self.play(
            LaggedStart(
                *[Create(obj) for obj in basis_func],
                run_time=4,
                lag_ratio=0.75,
            )
        )

        phi_cam, theta_cam, gamma_cam = get_euler_angles_from_rotation_matrix(
            look_at(ORIGIN, self.rotated_camera_position, np.array([0, 1, 0]))
        )

        self.play(Write(z_label_axes_coeffs), Write(z_label_axes_func))
        self.add(*z_tick_labels_all)

        self.move_camera(
            phi=phi_cam,
            theta=theta_cam,
            gamma=gamma_cam,
        )

        # self.play(
        #     *[
        #         z_tracker.animate.set_value(freq * delta_z)
        #         for freq, z_tracker in zip(range(-3, 4), z_trackers)
        #     ],
        # )
        # self.play(*[Create(line) for line in x_reference_lines])
        self.wait()
        # self.play(
        #     *[
        #         opacity_tracker.animate.set_value(1)
        #         if freq <= 0
        #         else opacity_tracker.animate.set_value(0.5)
        #         for freq, opacity_tracker in zip(range(-3, 4), opacity_trackers)
        #     ],
        # )
        #
        # print([z_tracker.get_value() for z_tracker in z_trackers])

        # self.play(LaggedStart(*[FadeIn(object) for object in coeff_dots + coeff_lines]))
        self.add(*coeff_dots, *coeff_lines)
        self.wait()
        np.random.seed(42)
        random_magnitudes = -1 + 2 * np.random.random(7)
        random_thetas = -np.pi + 2 * np.pi * np.random.random(7)
        # self.play(
        #     *[
        #         magnitude.animate.set_value(val)
        #         for val, magnitude in zip(random_magnitudes, magnitude_trackers)
        #     ],
        #     *[
        #         theta.animate.set_value(val)
        #         for val, theta in zip(random_thetas, theta_trackers)
        #     ],
        #     run_time=3,
        # )

        # self.play(
        #     *[
        #         magnitude.animate.set_value(val)
        #         for val, magnitude in zip(coeff_magnitudes, magnitude_trackers)
        #     ],
        #     *[
        #         theta.animate.set_value(val)
        #         for val, theta in zip(coeff_thetas, theta_trackers)
        #     ],
        #     run_time=3,
        # )
        self.wait(2)

        graph_fourier_approx = axes_func.get_graph(
            lambda t: np.real(
                np.sum(
                    [
                        fourier_coeffs[n + n_fourier_coeffs]
                        * np.exp(1j * t * 2 * np.pi * n)
                        for n in range(-n_fourier_coeffs, n_fourier_coeffs + 1)
                    ]
                )
            ),
            x_range=[-1.5, 1.5, 1e-3],
            color=ORANGE,
        )
        # self.play(
        #     *[
        #         ReplacementTransform(basis, graph_fourier_approx)
        #         for basis in basis_func
        #     ],
        #     *[
        #         z_tracker.animate.set_value(0)
        #         for freq, z_tracker in zip(range(-3, 4), z_trackers)
        #     ],
        # )
        # self.remove(*basis_func)
        # self.remove(*coeff_lines)
        # self.add(graph_fourier_approx)
        # self.move_camera(
        #     phi=0,
        #     theta=-90 * DEGREES,
        #     gamma=0,
        # )
        # self.wait(10)


class Intro(Scene):
    def construct(self):
        text_1 = Tex(r"Goal: A more intuitive understanding of the Fourier tranform:")
        fourier_transform = MathTex(
            r"{{[\mathcal F(f)](\omega)}}=\frac{1}{\sqrt{2\pi} }\int_{-\infty}^{+\infty} f(t) e^{-i\omega t}\mathrm d t",
            color=BLUE,
        )
        text_2 = Tex(
            r"Why are a differential equations transformed into algebraic equations by the Fourier transform?"
        )
        diff_eq = MathTex(r"m{{D^2}} f(t) + c {{D}}f(t) + kf(t) = F(t)", color=BLUE)
        diff_eq_2 = MathTex(
            r"m{{\frac{\mathrm d^2}{\mathrm d x^2}}} f(t) + c {{\frac{\mathrm d}{\mathrm dx} }}f(t) + kf(t) = F(t)",
            color=BLUE,
        )
        alg_eq = MathTex(
            r"-m\omega^2{{[\mathcal F(f)](\omega)}}+ic\omega{{[\mathcal F(f)](\omega)}} + k{{[\mathcal F(f)](\omega)}}={{[\mathcal F(F)](\omega)}}",
            color=BLUE,
        )
        sol = MathTex(
            r"{{f(t)}} =\frac{1}{\sqrt{2\pi} }\int_{-\infty}^{+\infty}{{[\mathcal F(f)](\omega)}}e^{i\omega t}\mathrm d \omega",
            color=BLUE,
        )
        text_3 = Tex(
            r"Why the use of {{$e^{-i\omega t}$}} as the kernel of the tranformation? How can this be interpreted?"
        )
        text_3.get_part_by_tex(r"$e^{-i\omega t}$").set_color(BLUE)
        full_group = VGroup(*[text_1, fourier_transform, text_2, diff_eq, text_3])

        full_group.arrange(DOWN, buff=std_line_buff)
        full_group.center()
        alg_eq.move_to(diff_eq.get_center())
        diff_eq_2.move_to(diff_eq.get_center())
        sol.move_to(diff_eq_2.get_center())

        self.wait(5)
        # Hi, in this video, we will try to improve our understanding of the Fourier
        # transform.
        self.play(Write(text_1), Write(fourier_transform), run_time=2)

        self.wait(3)
        # The transform is defined like so.
        self.play(Indicate(fourier_transform, color=ORANGE))

        self.wait(3)
        # This will be mainly focused on two questions.
        # First, why does the Fourier transform turn differential equations into algebraic
        # equations?
        self.play(Write(text_2), Write(diff_eq), run_time=2)
        self.wait(8)

        self.wait(1)
        # Here $D$ is differential operator and $f$ some unknown function depending on $t$.
        self.play(
            ReplacementTransform(diff_eq, diff_eq_2),
            run_time=2,
        )

        # Applying the Fourier transform, we get an algebraic equation, where there is a new
        # unknown function, the Fourier transform of $f$, $\mathcal F(f)$ function of $w$.
        # But no differentiation operators are in sight!
        self.wait(1)
        self.play(ReplacementTransform(diff_eq_2, alg_eq), run_time=2)
        self.play(
            Indicate(alg_eq.get_parts_by_tex(r"[\mathcal F(f)](\omega)"), color=ORANGE)
        )

        # We can now solve this algebraic equation and use the inverse Fourier transform to
        # find our initial unknown function $f$.
        self.wait(1)
        self.play(ReplacementTransform(alg_eq, sol), run_time=2)
        self.wait(1)
        self.play(Indicate(sol.get_part_by_tex(r"f(t)"), color=ORANGE))
        self.wait(1)

        # Second, and closely connected to the first, where does the kernel of the
        # transformation, i.e., $e^{-iwt}$, come from? How can we interpret it?
        self.play(Write(text_3), run_time=2)
        self.wait(2)

        self.play(
            *[
                Unwrite(mobject)
                for mobject in [text_1, fourier_transform, text_2, sol, text_3]
            ]
        )
        self.wait(2)

        # text_5 = Tex(r"We are going the analogy between ordinary vectors and functions as a starting point.")


class Outro(Scene):
    def construct(self):
        text_2 = Tex(
            r"Why are a differential equations transformed into algebraic equations by the Fourier transform?"
        )
        text_3 = Tex(
            r"Why the use of {{$e^{-iwt}$}} as the kernel of the tranformation? How can this be interpreted?"
        )
        text_3.get_part_by_tex(r"$e^{-iwt}$").set_color(BLUE)
        full_group = VGroup(*[text_2, text_3])

        full_group.arrange(DOWN, buff=3)
        full_group.center()

        self.play(Write(text_2), Write(text_3), run_time=2)
        self.wait(2)

        self.play(*[Unwrite(mobject) for mobject in [text_2, text_3]])
        self.wait(2)

        # text_5 = Tex(r"We are going the analogy between ordinary vectors and functions as a starting point.")


class FourierTransformExplainer(ThreeDScene):
    def setup_this_scene(self):

        self.rotated_camera_position = [2, 1.5, 1]
        self.rotated_camera_direction = self.rotated_camera_position / np.linalg.norm(
            self.rotated_camera_position
        )
        self.amplitude = 1.5

    def func_to_model(self, x, period=1):

        val = np.exp(-self.amplitude * x) if x > 0 else 0

        return val

    def fourier_transform(self, w):

        val = 1 / (self.amplitude + 2 * np.pi * 1j * w)

        return val

    def other_transform(self, w):

        val = 2 * self.amplitude / (self.amplitude ** 2 + 4 * np.pi ** 2 * w ** 2)

        return val

    def interpolate_func_coeff(self, w, t):

        val_1 = self.fourier_transform(w)
        val_2 = self.other_transform(w)

        return (1 - t) * val_1 + t * val_2

    def construct(self):

        self.setup_this_scene()
        delta_z = 1
        n_points = 500

        axes_func = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-4, 4, 1],
            # z_range=[-1e-4, 1e-4, 1],
            x_length=8,
            y_length=5,
            z_length=2.5 * 6,
            axis_config={"include_ticks": False},
            tips=False,
        )
        axes_coeffs = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-4, 4, 1],
            x_length=4,
            y_length=4,
            z_length=2.5 * 6,
            # z_axis_config={
            #     "numbers_to_include": list(range(-3, 4)),
            #     "label_direction": RIGHT,
            # },
            # x_axis_config={
            #     "numbers_to_include": [0.5],
            #     "label_direction": np.array([0.45]),
            # },
            # y_axis_config={
            #     "numbers_to_include": [0.5],
            #     "label_direction": np.array([0.45]),
            # },
            tips=False,
        )

        graph_x_values = np.linspace(-1.5, 1.5, 100)
        graph_y_values = [self.func_to_model(x) for x in graph_x_values]
        graph_z_values = [0 for _ in graph_x_values]

        graph = axes_func.get_line_graph(
            x_values=graph_x_values,
            y_values=graph_y_values,
            z_values=graph_z_values,
            add_vertex_dots=False,
        ).set_stroke(color=RED)

        all_func = VGroup(axes_func, graph)
        all_coeff = VGroup(axes_coeffs)
        both_graphs = VGroup(all_func, all_coeff)
        both_graphs.arrange(RIGHT)
        both_graphs.center()

        x_label = axes_coeffs.get_x_axis_label(r"\mathrm{Re}")
        x_label.move_to(axes_coeffs.coords_to_point(2, 0.5, 0))

        y_label = axes_coeffs.get_y_axis_label(r"\mathrm{Im}")
        y_label.move_to(axes_coeffs.coords_to_point(0.5, 2, 0))

        z_label = MathTex("\omega")
        z_label.move_to(axes_coeffs.coords_to_point(0.5, 0, 2))
        z_label_graph = MathTex("\omega")
        z_label_graph.move_to(axes_func.coords_to_point(0.5, 0, 3.5))

        z_tick_labels_base = []
        for ind in range(-3, 4):
            if ind == 0:
                continue

            z_tick_labels_base.append(MathTex("{0:d}".format(ind), font_size=8.0))
            z_tick_labels_base[-1].move_to(
                axes_coeffs.coords_to_point(-0.2, 0.2, delta_z * ind)
            )
            z_tick_labels_base[-1].scale(0.6)

        x_label_axes_coeffs = always_redraw(redraw_facing_camera(x_label, self))
        y_label_axes_coeffs = always_redraw(redraw_facing_camera(y_label, self))
        z_label_axes_coeffs = always_redraw(redraw_facing_camera(z_label, self))
        z_label_axes_func = always_redraw(redraw_facing_camera(z_label_graph, self))
        z_tick_labels_all = [
            always_redraw(redraw_facing_camera(tick, self))
            for tick in z_tick_labels_base
        ]

        all_coeff.add(x_label_axes_coeffs, y_label_axes_coeffs)

        self.add(all_func, all_coeff)
        # self.play(
        #     LaggedStart(
        #         *[
        #             Create(obj)
        #             for obj in [
        #                 axes_func,
        #                 graph,
        #                 axes_coeffs,
        #                 x_label_axes_coeffs,
        #                 y_label_axes_coeffs,
        #             ]
        #         ],
        #         run_time=4,
        #         lag_ratio=0.75,
        #     )
        # )

        x_reference_lines = []
        t = np.linspace(-2, 5, 10)
        for freq in range(-3, 4):
            x_values = t
            y_values = 10 * [0]
            z_values = 10 * [delta_z * freq]

            x_reference_lines.append(
                Line(
                    axes_func.coords_to_point(*[x_values[0], y_values[0], z_values[0]]),
                    axes_func.coords_to_point(
                        *[x_values[-1], y_values[-1], z_values[-1]]
                    ),
                )
            )
            if freq > 0:
                x_reference_lines[-1].set_stroke(color=BLUE, opacity=0.5, width=1.5)
            else:
                x_reference_lines[-1].set_stroke(color=BLUE, opacity=0.8, width=1.5)

        # The camera is auto set to PHI = 0 and THETA = -90
        coeff_dots = []
        coeff_lines = []
        magnitude_trackers = []
        theta_trackers = []
        z_trackers = []

        func_switcher = ValueTracker(1)

        for i_freq, freq in enumerate(range(-3, 4)):

            def create_dot_line_generator(freq):
                z_tracker = ValueTracker(0)
                magnitude_val = np.abs(
                    self.interpolate_func_coeff(freq, func_switcher.get_value())
                )

                theta_val = np.angle(
                    self.interpolate_func_coeff(freq, func_switcher.get_value())
                )

                def dot_generator():
                    return Dot3D(
                        point=axes_coeffs.coords_to_point(
                            np.real(magnitude_val * np.exp(1j * theta_val)),
                            np.imag(magnitude_val * np.exp(1j * theta_val)),
                            z_tracker.get_value(),
                        ),
                        color=DARK_BLUE,
                    )

                def line_generator():
                    return DashedLine(
                        start=axes_coeffs.coords_to_point(
                            0,
                            0,
                            z_tracker.get_value(),
                        ),
                        end=axes_coeffs.coords_to_point(
                            np.real(magnitude_val * np.exp(1j * theta_val)),
                            np.imag(magnitude_val * np.exp(1j * theta_val)),
                            z_tracker.get_value(),
                        ),
                    )

                return (
                    magnitude_val,
                    theta_val,
                    z_tracker,
                    dot_generator,
                    line_generator,
                )

            (
                current_magnitude_tracker,
                current_theta_tracker,
                current_z_tracker,
                dot_generator,
                line_generator,
            ) = create_dot_line_generator(freq)

            current_dot = always_redraw(dot_generator)
            current_line = always_redraw(line_generator)

            magnitude_trackers.append(current_magnitude_tracker)
            theta_trackers.append(current_theta_tracker)
            z_trackers.append(current_z_tracker)

            coeff_dots.append(current_dot)
            coeff_lines.append(current_line)

            # coeff_dots[-1].set_stroke(color=DARK_BLUE, width=0.3)
            # coeff_dots[-1].set_stroke(color=DARK_BLUE)

        # print([coeff_dot.get_center()[2] for coeff_dot in coeff_dots])

        opacity_trackers = [ValueTracker(1) for _ in range(-3, 4)]
        basis_func = []
        t = np.linspace(-1.5, 1.5, n_points)
        for freq in range(-3, 4):

            def create_func_generator(freq, axes):
                def func_generator():
                    x_values = t
                    y_values = np.real(
                        np.abs(
                            self.interpolate_func_coeff(freq, func_switcher.get_value())
                        )
                        * np.exp(
                            1j
                            * np.angle(
                                self.interpolate_func_coeff(
                                    freq, func_switcher.get_value()
                                )
                            )
                            + 1j * freq * t * 2 * np.pi / 1
                        )
                    )
                    z_values = n_points * [z_trackers[freq + 3].get_value()]
                    func = axes.get_line_graph(
                        x_values=x_values,
                        y_values=y_values,
                        z_values=z_values,
                        line_color=BLUE,
                        add_vertex_dots=False,
                    )
                    func.set_stroke(
                        color=DARK_BLUE,
                        opacity=opacity_trackers[freq + 3].get_value(),
                    )
                    # if freq > 0:
                    #     func.set_stroke(
                    #         color=DARK_BLUE,
                    #         opacity=opacity_trackers[freq + 3].get_value(),
                    #         width=0.5,
                    #     )
                    # else:
                    #     func.set_stroke(
                    #         color=DARK_BLUE,
                    #         opacity=opacity_trackers[freq + 3].get_value(),
                    #     )
                    return func

                return func_generator

            basis_func.append(
                always_redraw(
                    create_func_generator(
                        freq,
                        axes_func,  # , magnitude_trackers, theta_trackers, z_trackers
                    )
                )
            )

        self.wait()
        self.add(*basis_func)
        # self.play(
        #     LaggedStart(
        #         *[Create(obj) for obj in basis_func],
        #         run_time=4,
        #         lag_ratio=0.75,
        #     )
        # )

        self.play(Write(z_label_axes_coeffs), Write(z_label_axes_func))
        self.add(*z_tick_labels_all)

        phi_cam, theta_cam, gamma_cam = get_euler_angles_from_rotation_matrix(
            look_at(ORIGIN, self.rotated_camera_position, np.array([0, 1, 0]))
        )
        self.move_camera(
            phi=phi_cam,
            theta=theta_cam,
            gamma=gamma_cam,
        )

        def surface_generator():
            surf = ParametricSurface(
                lambda u, v: axes_func.c2p(
                    u,
                    np.real(
                        self.interpolate_func_coeff(v, func_switcher.get_value())
                        * np.exp(1j * v * u * 2 * np.pi / 1)
                    ),
                    v,
                ),
                u_min=-1.5,
                u_max=1.5,
                v_min=-4,
                v_max=4,
                resolution=(30, 20),
            )
            surf.set_style(
                fill_opacity=0.3,
                stroke_color=WHITE,
                stroke_width=0.5,
                stroke_opacity=0.5,
            )
            return surf

        surface = always_redraw(surface_generator)

        # surface.set_fill_by_value(
        #     axes=axes_coeffs, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)]
        # )
        self.add(surface)

        graph2 = always_redraw(
            lambda: axes_coeffs.get_parametric_curve(
                lambda w: np.array(
                    [
                        np.real(
                            self.interpolate_func_coeff(w, func_switcher.get_value())
                        ),
                        np.imag(
                            self.interpolate_func_coeff(w, func_switcher.get_value())
                        ),
                        w,
                    ]
                ),
                t_range=[-4, 4],
                color=DARK_BLUE,
            )
        )
        self.add(graph2)

        self.play(
            *[
                z_tracker.animate.set_value(freq * delta_z)
                for freq, z_tracker in zip(range(-3, 4), z_trackers)
            ],
        )
        self.play(func_switcher.animate.set_value(0), run_time=4)
        # # self.play(*[Create(line) for line in x_reference_lines])
        # self.wait()
        # # self.play(
        # #     *[
        # #         opacity_tracker.animate.set_value(1)
        # #         if freq <= 0
        # #         else opacity_tracker.animate.set_value(0.5)
        # #         for freq, opacity_tracker in zip(range(-3, 4), opacity_trackers)
        # #     ],
        # # )
        # #
        # # print([z_tracker.get_value() for z_tracker in z_trackers])
        #
        # # self.play(LaggedStart(*[FadeIn(object) for object in coeff_dots + coeff_lines]))
        # self.add(*coeff_dots, *coeff_lines)
        # self.wait()
        # np.random.seed(42)
        # random_magnitudes = -1 + 2 * np.random.random(7)
        # random_thetas = -np.pi + 2 * np.pi * np.random.random(7)
        # # self.play(
        # #     *[
        # #         magnitude.animate.set_value(val)
        # #         for val, magnitude in zip(random_magnitudes, magnitude_trackers)
        # #     ],
        # #     *[
        # #         theta.animate.set_value(val)
        # #         for val, theta in zip(random_thetas, theta_trackers)
        # #     ],
        # #     run_time=3,
        # # )
        #
        # # self.play(
        # #     *[
        # #         magnitude.animate.set_value(val)
        # #         for val, magnitude in zip(coeff_magnitudes, magnitude_trackers)
        # #     ],
        # #     *[
        # #         theta.animate.set_value(val)
        # #         for val, theta in zip(coeff_thetas, theta_trackers)
        # #     ],
        # #     run_time=3,
        # # )
        # self.wait(2)
        #
        # graph_fourier_approx = axes_func.get_graph(
        #     lambda t: np.real(
        #         np.sum(
        #             [
        #                 fourier_coeffs[n + n_fourier_coeffs]
        #                 * np.exp(1j * t * 2 * np.pi * n)
        #                 for n in range(-n_fourier_coeffs, n_fourier_coeffs + 1)
        #             ]
        #         )
        #     ),
        #     x_range=[-1.5, 1.5, 1e-3],
        #     color=ORANGE,
        # )
        # # self.play(
        # #     *[
        # #         ReplacementTransform(basis, graph_fourier_approx)
        # #         for basis in basis_func
        # #     ],
        # #     *[
        # #         z_tracker.animate.set_value(0)
        # #         for freq, z_tracker in zip(range(-3, 4), z_trackers)
        # #     ],
        # # )
        # # self.remove(*basis_func)
        # # self.remove(*coeff_lines)
        # # self.add(graph_fourier_approx)
        # # self.move_camera(
        # #     phi=0,
        # #     theta=-90 * DEGREES,
        # #     gamma=0,
        # # )
        self.wait(10)


class TestScene(Scene):
    def construct(self):
        title = Tex("Space", " of all ", "triangles")
        title.scale(1.5)
        title.to_edge(UP)

        question = Tex("What ", "is ", "a\\\\", "moduli ", "space", "?")
        question.scale(2)

        self.play(
            LaggedStartMap(FadeIn, question),
        )
        self.wait()

        self.play(
            ReplacementTransform(
                question.get_part_by_tex("space"),
                title.get_part_by_tex("Space"),
            ),
            FadeOut(question[:-2]),
            FadeOut(question[-1]),
            FadeIn(title[1:]),
        )

        self.wait()


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


def look_at(view_target, camera_position, view_up):

    camera_direction = np.array(camera_position) - np.array(view_target)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    camera_right = np.cross(camera_direction, view_up)
    camera_right = camera_right / np.linalg.norm(camera_right)

    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rot_mat = np.array([camera_right, camera_up, camera_direction])

    return rot_mat


def get_euler_angles_from_rotation_matrix(rot_mat):
    """
    Get Euler rotation angles from rotation matrix.

    Angles according to Z-X-Z convention and manim's `move_camera`.
    """

    phi_cam = np.arctan2(rot_mat[2, 0], rot_mat[2, 1])
    theta_cam = np.arccos(rot_mat[2, 2])
    gamma_cam = -np.arctan2(rot_mat[0, 2], rot_mat[1, 2])

    return -phi_cam, -theta_cam - 90 * DEGREES, gamma_cam


def redraw_facing_camera(object, scene):
    """Function to achieve "track to camera" or "billboard" effect."""

    def redraw_func():
        new_obj = object.copy()
        new_obj.apply_matrix(
            scene.camera.get_rotation_matrix().T,
            about_point=new_obj.get_center(),
        )
        return new_obj

    return redraw_func


def create_own_replacement_transform(full_source, source_ind, replacement):
    to_replace = full_source[source_ind]
    to_the_left = full_source[:source_ind]
    to_the_right = full_source[source_ind + 1 :]
    replacement.move_to(to_replace.get_center())
    full_group = VGroup(to_the_left, to_replace, to_the_right)
    # full_group.arrange(RIGHT)
    # to_replace.match_height(to_the_right)
    transform = Transform(to_replace, replacement)
    # rearrange = full_group.animate.arrange(RIGHT)

    return transform  # , rearrange
